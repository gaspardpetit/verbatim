"""Repo-local dataset runner for SwitchLingua-style evaluation manifests."""
# pylint: disable=too-many-lines,wrong-import-position

import argparse
import ast
import csv
import gc
import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPT_DIR = Path(__file__).resolve().parent
_BENCHMARK_ROOT = _SCRIPT_DIR.parent
_DEFAULT_OUTDIR = _BENCHMARK_ROOT / "out"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from systems import SYSTEMS_CONFIG_PATH, load_system_specs  # noqa: E402

from tools._baseline_common import load_audio_mono, write_outputs  # noqa: E402
from verbatim.eval.compare import Metrics, compute_metrics  # noqa: E402
from verbatim.languages import normalize_language  # noqa: E402
from verbatim.models import Models  # noqa: E402
from verbatim.transcript.words import Utterance  # noqa: E402
from verbatim_benchmarks.normalizers import BasicTextNormalizer  # noqa: E402
from verbatim_cli.args import build_parser  # noqa: E402
from verbatim_cli.configure import build_output_formats, build_prefixes, compute_log_level, make_config  # noqa: E402
from verbatim_cli.env import load_env_file  # noqa: E402
from verbatim_cli.run_single import run_single_input  # noqa: E402
from verbatim_files.format.json import read_dlm_utterances, read_utterances  # noqa: E402

LOG = logging.getLogger(__name__)
LOGOGRAPHIC_LANGS = {"zh", "yue", "ja", "zh-hans", "zh-hant"}
NONLEXICAL_UTTERANCE_RE = re.compile(r"^\[[^\]]+\](?:\s+\[[^\]]+\])*$")
_BASIC_NORMALIZER = BasicTextNormalizer()
DEFAULT_SYSTEMS = load_system_specs()

PATH_FIELDS = ("audio", "audio_path", "audio_file", "path", "file", "filename")
REFERENCE_FIELDS = ("reference", "reference_path", "reference_file", "ref", "ref_path", "ref_file")
ID_FIELDS = ("id", "item", "item_id", "sample_id", "audio_id", "uid")
TEXT_LIST_FIELDS = ("data_generation_result", "instances", "reference_texts", "utterances", "texts")
TEXT_FIELDS = ("text", "transcript", "reference_text", "sentence")
METADATA_KEYS = (
    "topic",
    "tense",
    "perspective",
    "cs_ratio",
    "gender",
    "age",
    "education_level",
    "first_language",
    "second_language",
    "conversation_type",
    "cs_function",
    "cs_type",
    "language_pair",
)
LANGUAGE_FOLDER_MAP = {
    "arabic": "Arabic",
    "cantonese": "Cantonese",
    "french": "French",
    "german": "German",
    "hindi": "Hindi",
    "italian": "Italian",
    "japanese": "Japanese",
    "korean": "Korean",
    "mandarin": "Mandarin",
    "russian": "Russian",
    "spanish": "Spanish",
    "english": "English",
}


@dataclass
class DatasetItem:
    item_id: str
    audio_path: Path
    reference_utterances: List[Utterance]
    normalized_reference_text: Optional[str]
    languages: List[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    item_id: str
    system: str
    status: str
    audio_path: str
    output_json: Optional[str]
    elapsed_seconds: float
    languages: List[str]
    wer: Optional[float] = None
    wder: Optional[float] = None
    cpwer: Optional[float] = None
    spkcntmae: Optional[float] = None
    cer: Optional[float] = None
    sem: Optional[float] = None
    saer: Optional[float] = None
    wer_sub: Optional[int] = None
    wer_del: Optional[int] = None
    wer_ins: Optional[int] = None
    wer_total: Optional[int] = None
    cer_dist: Optional[int] = None
    cer_total: Optional[int] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class WhisperBaselineRunner:
    def __init__(self, *, model_name: str, device: str):
        try:
            import whisper  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("Official Whisper baseline requires the optional `openai-whisper` package.") from exc

        self._model = whisper.load_model(model_name, device=device)
        self._device = device
        self._model_name = model_name

    def transcribe(self, audio_path: Path, output_dir: Path, stem: str, forced_language: Optional[str] = None) -> tuple[Path, dict[str, Any]]:
        audio = load_audio_mono(audio_path)
        result = self._model.transcribe(
            audio,
            language=forced_language,
            task="transcribe",
            verbose=False,
            word_timestamps=False,
            condition_on_previous_text=False,
            fp16=self._device == "cuda",
        )
        segments = result.get("segments", []) or []
        utterances = [
            Utterance(
                utterance_id=f"utt{index}",
                speaker="SPEAKER",
                start_ts=int(float(segment.get("start", 0.0)) * 16000),
                end_ts=int(float(segment.get("end", 0.0)) * 16000),
                text=(segment.get("text", "") or "").strip(),
                words=[],
            )
            for index, segment in enumerate(segments, start=1)
            if (segment.get("text", "") or "").strip()
        ]
        if not utterances:
            utterances = [
                Utterance(
                    utterance_id="utt1",
                    speaker="SPEAKER",
                    start_ts=0,
                    end_ts=len(audio),
                    text=(result.get("text", "") or "").strip(),
                    words=[],
                )
            ]
        json_path, _ = write_outputs(
            outdir=output_dir,
            stem=stem,
            utterances=utterances,
            metadata={
                "baseline": {
                    "backend": "official",
                    "model": self._model_name,
                    "device": self._device,
                    "detected_language": result.get("language", "und") or "und",
                    "detected_language_probability": None,
                }
            },
        )
        return json_path, {"detected_language": result.get("language", "und") or "und"}


class QwenBaselineRunner:
    def __init__(self, *, model_name: str, device: str):
        if device not in ("cpu", "cuda", "mps"):
            raise RuntimeError("Qwen3-ASR baseline supports only 'cpu', 'cuda', and 'mps'.")
        try:
            import torch  # pylint: disable=import-outside-toplevel
            from qwen_asr import Qwen3ASRModel  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("Qwen3-ASR baseline requires optional dependencies. Install `qwen-asr` and `torch`.") from exc
        from verbatim.voices.transcribe.qwen_asr import QwenAsrTranscriber  # pylint: disable=import-outside-toplevel

        qwen_dtype = QwenAsrTranscriber._resolve_dtype(torch_module=torch, dtype="auto", device=device)
        self._helper = QwenAsrTranscriber
        self._model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=qwen_dtype,
            device_map="cuda:0" if device == "cuda" else device,
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        self._device = device
        self._model_name = model_name

    def _guess_language(self, audio, allowed_languages: list[str]) -> tuple[str, float]:
        if not allowed_languages:
            return "en", 1.0
        if len(allowed_languages) == 1:
            return allowed_languages[0], 1.0
        results = self._model.transcribe(
            audio=(audio, 16000),
            language=None,
            return_time_stamps=False,
        )
        if not results:
            return allowed_languages[0], 0.0
        language = self._helper._get_field(results[0], "language")
        detected = self._helper._language_code(language, allowed_languages)
        return detected, 1.0 if detected in allowed_languages else 0.0

    def transcribe(
        self,
        audio_path: Path,
        output_dir: Path,
        stem: str,
        allowed_languages: list[str],
        forced_language: Optional[str] = None,
    ) -> tuple[Path, dict[str, Any]]:
        audio = load_audio_mono(audio_path)
        detected_language = forced_language
        detected_prob = 1.0 if forced_language else None
        if not detected_language:
            detected_language, detected_prob = self._guess_language(audio, allowed_languages)
        qwen_language = self._helper._language_name(detected_language)
        if qwen_language is None:
            raise RuntimeError(f"Language '{detected_language}' is not supported by Qwen3-ASR.")
        results = self._model.transcribe(
            audio=(audio, 16000),
            language=qwen_language,
            return_time_stamps=False,
        )
        text = ""
        if results:
            text = (self._helper._get_field(results[0], "text") or "").strip()
        utterances = [
            Utterance(
                utterance_id="utt1",
                speaker="SPEAKER",
                start_ts=0,
                end_ts=len(audio),
                text=text,
                words=[],
            )
        ]
        json_path, _ = write_outputs(
            outdir=output_dir,
            stem=stem,
            utterances=utterances,
            metadata={
                "baseline": {
                    "backend": "qwen",
                    "model": self._model_name,
                    "aligner_model": None,
                    "device": self._device,
                    "word_alignment": False,
                    "detected_language": detected_language,
                    "detected_language_probability": detected_prob,
                    "allowed_languages": allowed_languages,
                }
            },
        )
        return json_path, {"detected_language": detected_language}


def _slugify(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value.strip())
    return sanitized.strip("_") or "item"


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _canonicalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    for key, value in record.items():
        canonical = _canonicalize_key(key)
        if canonical and canonical not in normalized:
            normalized[canonical] = value
    return normalized


def _normalize_language(value: Any) -> Optional[str]:
    normalized = normalize_language(value)
    if normalized:
        return normalized
    LOG.debug("Failed to normalize language %r", value)
    return None


def _parse_languages(record: dict[str, Any]) -> List[str]:
    raw = record.get("languages")
    if isinstance(raw, list):
        languages = [_normalize_language(value) for value in raw]
        return [lang for lang in languages if lang]
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.replace("/", ",").replace("|", ",").split(",")]
        languages = [_normalize_language(part) for part in parts if part]
        return [lang for lang in languages if lang]

    pair_value = _first_present(record, ("language_pair", "lang_pair"))
    if isinstance(pair_value, str):
        normalized: List[str] = []
        separators = ("-", "/", "|", ",")
        tokens = [pair_value]
        for separator in separators:
            if separator in pair_value:
                tokens = [part.strip() for part in pair_value.split(separator)]
                break
        for token in tokens:
            language = _normalize_language(token)
            if language:
                normalized.append(language)
        if normalized:
            return normalized

    languages = []
    for key in ("first_language", "second_language", "matrix_language", "embedded_language"):
        language = _normalize_language(record.get(key))
        if language and language not in languages:
            languages.append(language)
    return languages


def _string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        stripped = value.strip()
        parsed_values = _extract_embedded_text_list(stripped)
        if parsed_values is not None:
            return parsed_values
    if isinstance(value, list):
        values = []
        for entry in value:
            if isinstance(entry, dict):
                text_value = _first_present(entry, ("text", "utterance", "content", "ref_text"))
                if text_value not in (None, ""):
                    values.append(str(text_value).strip())
            elif isinstance(entry, str):
                parsed_values = _extract_embedded_text_list(entry.strip())
                if parsed_values is not None:
                    values.extend(parsed_values)
                elif entry not in (None, ""):
                    values.append(str(entry).strip())
            elif entry not in (None, ""):
                values.append(str(entry).strip())
        return [value for value in values if value]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _extract_embedded_text_list(value: str) -> Optional[List[str]]:
    if not value:
        return None
    candidate = value.strip()
    if not candidate or candidate[0] not in "[{":
        return None

    parsed = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(candidate)
            break
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    if parsed is None:
        return None

    if isinstance(parsed, dict):
        for key in ("instances", "data_generation_result", "texts", "utterances"):
            extracted = _string_list(parsed.get(key))
            if extracted:
                return extracted
        text_value = _first_present(parsed, ("text", "turn", "utterance", "content", "ref_text"))
        extracted = _string_list(text_value)
        return extracted or None

    if isinstance(parsed, list):
        extracted = _string_list(parsed)
        return extracted or None

    return None


def _reference_from_record(record: dict[str, Any]) -> List[Utterance]:
    text_values: List[str] = []
    for key in TEXT_LIST_FIELDS:
        if key in record:
            text_values = _string_list(record[key])
            if text_values:
                break
    if not text_values:
        for key in TEXT_FIELDS:
            if key in record:
                text_values = _string_list(record[key])
                if text_values:
                    break
    if not text_values:
        raise ValueError("Record does not define reference text or an explicit reference file.")

    return [
        Utterance(
            utterance_id=f"utt{index}",
            speaker=None,
            start_ts=0,
            end_ts=0,
            text=text,
            words=[],
        )
        for index, text in enumerate(text_values)
    ]


def _resolve_path(path_value: str, *, explicit_root: Optional[Path], manifest_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = explicit_root if explicit_root is not None else manifest_root
    return (base / path).resolve()


def _load_reference_file(path: Path) -> List[Utterance]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    utterances = data.get("utterances", [])
    if utterances and isinstance(utterances[0], dict) and "ref_text" in utterances[0]:
        return read_dlm_utterances(str(path))
    return read_utterances(str(path))


def load_manifest(
    path: Path,
    *,
    audio_root: Optional[Path],
    reference_root: Optional[Path],
    limit: Optional[int],
    default_languages: Optional[List[str]] = None,
    lang_filter: Optional[str] = None,
) -> List[DatasetItem]:
    manifest_root = path.parent.resolve()
    records = _load_records(path)
    items: List[DatasetItem] = []
    normalized_lang_filter = _normalize_language(lang_filter) if lang_filter else None
    for record in records:
        item_languages = _parse_languages(record)
        if normalized_lang_filter and item_languages and normalized_lang_filter not in item_languages:
            continue

        audio_value = _first_present(record, PATH_FIELDS)
        if not audio_value:
            raise ValueError(f"Record is missing an audio path field: expected one of {PATH_FIELDS}")

        item_id = str(_first_present(record, ID_FIELDS) or Path(str(audio_value)).stem)
        normalized_id = _slugify(item_id)
        audio_path = _resolve_path(str(audio_value), explicit_root=audio_root, manifest_root=manifest_root)

        reference_value = _first_present(record, REFERENCE_FIELDS)
        if reference_value:
            reference_path = _resolve_path(str(reference_value), explicit_root=reference_root, manifest_root=manifest_root)
            reference_utterances = _load_reference_file(reference_path)
        else:
            reference_utterances = _reference_from_record(record)

        metadata = {key: record[key] for key in METADATA_KEYS if key in record and record[key] not in (None, "")}
        if not item_languages and default_languages:
            item_languages = list(default_languages)
        if normalized_lang_filter and item_languages and normalized_lang_filter not in item_languages:
            continue
        normalized_reference_text = str(record.get("normalized", "")).strip() or None

        items.append(
            DatasetItem(
                item_id=normalized_id,
                audio_path=audio_path,
                reference_utterances=reference_utterances,
                normalized_reference_text=normalized_reference_text,
                languages=item_languages,
                metadata=metadata,
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def _load_records(path: Path) -> List[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [_canonicalize_record(dict(row)) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [_canonicalize_record(json.loads(line)) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [_canonicalize_record(item) if isinstance(item, dict) else item for item in payload]
        if isinstance(payload, dict):
            for key in ("items", "records", "data", "instances"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [_canonicalize_record(item) if isinstance(item, dict) else item for item in value]
            return [_canonicalize_record(payload)]
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def build_switchlingua_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmarks/switchlingua/scripts/benchmark.py",
        description="Run Verbatim comparisons on a SwitchLingua-style manifest",
    )
    parser.add_argument("--manifest", required=False, help="Path to a JSON, JSONL, or CSV manifest describing dataset items")
    parser.add_argument("--lang", default=None, help="Convenience language name (e.g., french) to auto-select manifest/audio root/run-name")
    parser.add_argument("--audio-root", default=None, help="Optional root directory prepended to relative audio paths")
    parser.add_argument("--reference-root", default=None, help="Optional root directory prepended to relative reference paths")
    parser.add_argument("--systems-config", default=str(SYSTEMS_CONFIG_PATH), help="YAML file defining benchmark systems")
    parser.add_argument("--systems", nargs="*", default=["qwen_mms", "qwen_mms_naive"], help="Named systems to run")
    parser.add_argument("--list-systems", action="store_true", help="List the available system presets and exit")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of manifest items to process")
    parser.add_argument("--txt", action="store_true", help="Also emit TXT output for each run")
    parser.add_argument("--json", action="store_true", help="Emit JSON output for each run (enabled by default)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs when the output JSON already exists")
    parser.add_argument("--continue-on-error", action="store_true", help="Record run failures and continue with the remaining items")
    parser.add_argument("-o", "--outdir", default=str(_DEFAULT_OUTDIR), help="Base output directory for run artifacts and summaries")
    parser.add_argument("--run-name", default=None, help="Optional subdirectory name under outdir (e.g. 'arabic')")
    parser.add_argument("-w", "--workdir", default=None, help="Base working directory for intermediate artifacts")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    parser.add_argument("--offline", action="store_true", help="Disallow downloads and require cached models")
    parser.add_argument("--model-cache", default=None, help="Deterministic cache directory for downloaded models")
    parser.add_argument(
        "--language-detection-initial-seconds", type=float, default=None, help="Initial audio duration in seconds for language detection"
    )
    parser.add_argument(
        "--language-detection-increment-seconds", type=float, default=None, help="Additional seconds added at each language detection retry"
    )
    parser.add_argument(
        "--language-detection-factor", type=float, default=None, help="Multiplicative factor applied at each language detection retry"
    )
    parser.add_argument("--mms-lid-model-size", default=None, help="Override MMS LID model identifier")
    parser.add_argument("--whisper-model", default=None, help="Override Whisper model size/path for whisper-based systems")
    parser.add_argument("--voxtral-model", default=None, help="Override Voxtral model identifier")
    parser.add_argument("--voxtral-max-new-tokens", type=int, default=None, help="Override Voxtral max_new_tokens")
    parser.add_argument("--qwen-model", default=None, help="Override Qwen3-ASR model identifier")
    parser.add_argument("--qwen-aligner-model", default=None, help="Override Qwen forced aligner model identifier")
    parser.add_argument("--saer", action="store_true", help="Compute semantic metrics (SEM/SAER) using LaBSE embeddings")
    parser.add_argument("--saer-model", default="sentence-transformers/LaBSE", help="Sentence-transformers model for SEM/SAER")
    parser.add_argument("--saer-alpha", type=float, default=0.5, help="SAER alpha weight (0..1)")
    parser.add_argument("--saer-device", default=None, help="Device for SEM/SAER model (e.g., cpu, cuda)")
    parser.add_argument(
        "--install-switchlingua",
        action="store_true",
        help="Clone SwitchLingua into ext/switchlingua if it is missing (non-submodule).",
    )
    parser.add_argument(
        "--diarize",
        choices=["pyannote", "senko", "energy", "channel", "separate"],
        default=None,
        help="Optional diarization strategy",
    )
    parser.add_argument("--speakers", nargs="?", default=None, help="Optional speaker count hint")
    return parser


def _resolve_lang_defaults(lang: str) -> tuple[Path, Path, str]:
    normalized = lang.strip().lower()
    if not normalized:
        raise ValueError("Language name cannot be empty.")
    manifest_path = Path("benchmarks") / "switchlingua" / "manifests" / "manifest_bootstrap.jsonl"
    audio_root = Path("ext") / "switchlingua" / "SwitchLingua_audio"
    run_name = normalized
    return manifest_path, audio_root, run_name


def _default_languages_for_lang(lang: Optional[str]) -> Optional[List[str]]:
    if not lang:
        return None
    primary = _normalize_language(lang)
    if not primary:
        return None
    if primary == "en":
        return ["en"]
    return [primary, "en"]


def list_systems() -> None:
    for name, spec in DEFAULT_SYSTEMS.items():
        print(f"{name}: {spec['description']}")


def _system_mode(system_name: str) -> str:
    return str(DEFAULT_SYSTEMS[system_name].get("mode", "pipeline"))


def _fixed_primary_language(system_name: str) -> bool:
    return bool(DEFAULT_SYSTEMS[system_name].get("fixed_primary_language", False))


def _create_shared_runner(args: argparse.Namespace, system_name: str) -> Optional[Any]:
    mode = _system_mode(system_name)
    device = "cpu" if args.cpu else "cuda"
    if mode == "whisper_baseline":
        return WhisperBaselineRunner(model_name=args.whisper_model or "large-v3", device=device)
    if mode == "qwen_baseline":
        return QwenBaselineRunner(model_name=args.qwen_model or "Qwen/Qwen3-ASR-1.7B", device=device)
    return None


def _ensure_switchlingua_repo(*, install: bool) -> None:
    repo_path = Path("ext") / "switchlingua"
    if repo_path.exists():
        return
    if not install:
        raise FileNotFoundError(
            "SwitchLingua repo not found at ext/switchlingua. "
            "Clone it with: git clone https://github.com/Shelton1013/SwitchLingua ext/switchlingua "
            "or re-run with --install-switchlingua."
        )
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Cloning SwitchLingua into %s", repo_path)
    subprocess.run(
        ["git", "clone", "https://github.com/Shelton1013/SwitchLingua", str(repo_path)],
        check=True,
    )


def _base_run_args() -> Namespace:
    parser = build_parser(prog="switchlingua-run")
    return parser.parse_args([])


def _build_run_namespace(
    args: argparse.Namespace,
    *,
    item: DatasetItem,
    system_name: str,
    item_output_dir: Path,
    item_work_dir: Optional[Path],
) -> Namespace:
    if system_name not in DEFAULT_SYSTEMS:
        raise ValueError(f"Unknown system: {system_name}")

    run_args = _base_run_args()
    run_args.input = str(item.audio_path)
    run_args.outdir = str(item_output_dir)
    run_args.workdir = str(item_work_dir) if item_work_dir is not None else None
    run_args.languages = item.languages or ["en"]
    run_args.verbose = args.verbose
    run_args.cpu = args.cpu
    run_args.offline = args.offline
    run_args.model_cache = args.model_cache
    run_args.json = True
    run_args.txt = args.txt
    run_args.quiet = True
    run_args.stdout_nocolor = True
    run_args.eval = None
    run_args.diarize = args.diarize
    run_args.speakers = args.speakers

    for key, value in DEFAULT_SYSTEMS[system_name]["overrides"].items():
        setattr(run_args, key, value)

    if args.mms_lid_model_size:
        run_args.mms_lid_model_size = args.mms_lid_model_size
    if args.whisper_model:
        run_args.whisper_model = args.whisper_model
    if args.voxtral_model:
        run_args.voxtral_model = args.voxtral_model
    if args.voxtral_max_new_tokens is not None:
        run_args.voxtral_max_new_tokens = args.voxtral_max_new_tokens
    if args.language_detection_initial_seconds is not None:
        run_args.language_detection_initial_seconds = args.language_detection_initial_seconds
    if args.language_detection_increment_seconds is not None:
        run_args.language_detection_increment_seconds = args.language_detection_increment_seconds
    if args.language_detection_factor is not None:
        run_args.language_detection_factor = args.language_detection_factor
    return run_args


def _apply_config_overrides(config, args: argparse.Namespace) -> None:
    if args.qwen_model:
        config.qwen_asr_model_size = args.qwen_model
    if args.voxtral_model:
        config.voxtral_model_size = args.voxtral_model
    if args.voxtral_max_new_tokens is not None:
        config.voxtral_max_new_tokens = args.voxtral_max_new_tokens
    if args.qwen_aligner_model:
        config.qwen_aligner_model_size = args.qwen_aligner_model


def _metrics_to_result(metrics: Metrics) -> dict[str, float]:
    return {
        "wer": float(metrics.WER),
        "wder": float(metrics.WDER),
        "cpwer": float(metrics.cpWER),
        "spkcntmae": float(metrics.SpkCntMAE),
    }


def _normalize_lang_code(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    return lang.lower().replace("_", "-")


def _form_normalizer(lang: Optional[str]):
    return _BASIC_NORMALIZER


def _normalize_form_text(text: str, lang: Optional[str]) -> str:
    normalized = _form_normalizer(lang)(text or "")
    return normalized.strip()


def _is_logographic(lang: Optional[str]) -> bool:
    normalized = _normalize_lang_code(lang)
    if normalized in LOGOGRAPHIC_LANGS:
        return True
    return bool(normalized and normalized.startswith("zh"))


def _edit_distance(a: List[str], b: List[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cost = 0
            else:
                cost = 1
            current.append(
                min(
                    prev[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    prev[j - 1] + cost,  # substitution
                )
            )
        prev = current
    return prev[-1]


def _cer(hyp_text: str, ref_text: str) -> Optional[float]:
    ref_chars = list(ref_text)
    if not ref_chars:
        return None
    distance = _edit_distance(list(hyp_text), ref_chars)
    return distance / len(ref_chars)


def _compute_corpus_counts(
    hyp_utterances: List[Utterance],
    ref_utterances: List[Utterance],
    lang_code: Optional[str],
    normalized_ref_text: Optional[str] = None,
) -> dict[str, Optional[int]]:
    try:
        from verbatim.eval.diarization_metrics import compute_utterance_metrics
        from verbatim.eval.diarization_utils import normalize_text
    except Exception:
        return {
            "wer_sub": None,
            "wer_del": None,
            "wer_ins": None,
            "wer_total": None,
            "cer_dist": None,
            "cer_total": None,
        }

    hyp_text = _normalize_form_text(_concat_text(hyp_utterances), lang_code)
    ref_text = normalized_ref_text if normalized_ref_text else _normalize_form_text(_concat_text(ref_utterances), lang_code)
    if not ref_text:
        return {
            "wer_sub": None,
            "wer_del": None,
            "wer_ins": None,
            "wer_total": None,
            "cer_dist": None,
            "cer_total": None,
        }

    metrics = compute_utterance_metrics(hyp_text=hyp_text, ref_text=ref_text)
    ref_norm = normalize_text(ref_text)
    hyp_norm = normalize_text(hyp_text)
    if ref_norm:
        cer_dist = _edit_distance(list(hyp_norm), list(ref_norm))
        cer_total = len(ref_norm)
    else:
        cer_dist = None
        cer_total = None

    return {
        "wer_sub": metrics.wer_sub,
        "wer_del": metrics.wer_delete,
        "wer_ins": metrics.wer_insert,
        "wer_total": metrics.wer_total,
        "cer_dist": cer_dist,
        "cer_total": cer_total,
    }


def _concat_text(utterances: List[Utterance]) -> str:
    filtered = _filter_scored_utterances(utterances)
    return " ".join(u.text.strip() for u in filtered if u.text and u.text.strip())


def _is_nonlexical_annotation_utterance(utterance: Utterance) -> bool:
    text = (utterance.text or "").strip()
    if not text:
        return False
    if utterance.words:
        return False
    return bool(NONLEXICAL_UTTERANCE_RE.fullmatch(text))


def _filter_scored_utterances(utterances: List[Utterance]) -> List[Utterance]:
    return [utterance for utterance in utterances if not _is_nonlexical_annotation_utterance(utterance)]


def _normalize_form_utterances(utterances: List[Utterance], lang_code: Optional[str]) -> List[Utterance]:
    normalized: List[Utterance] = []
    for utterance in _filter_scored_utterances(utterances):
        raw_text = (utterance.text or "").strip()
        if not raw_text:
            continue
        normalized_text = _normalize_form_text(raw_text, lang_code)
        if not normalized_text:
            continue
        normalized.append(
            Utterance(
                utterance_id=utterance.utterance_id,
                speaker=utterance.speaker,
                start_ts=utterance.start_ts,
                end_ts=utterance.end_ts,
                text=normalized_text,
                words=[],
            )
        )
    return normalized


def _write_score_text_artifacts(output_json_path: Path, utterances: List[Utterance], lang_code: Optional[str]) -> None:
    raw_text = _concat_text(utterances).strip()
    normalized_text = _normalize_form_text(raw_text, lang_code)
    txt_path = output_json_path.with_suffix(".txt")
    normalized_txt_path = output_json_path.with_name(f"{output_json_path.stem}.normal.txt")
    txt_path.write_text(raw_text, encoding="utf-8")
    normalized_txt_path.write_text(normalized_text, encoding="utf-8")


def _configure_model_cache_env(model_cache_dir: Optional[str], offline: bool) -> None:
    if offline:
        os.environ["VERBATIM_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.setdefault("VERBATIM_OFFLINE", "0")

    if not model_cache_dir:
        return

    os.makedirs(model_cache_dir, exist_ok=True)
    os.environ["VERBATIM_MODEL_CACHE"] = model_cache_dir

    xdg_cache = os.path.join(model_cache_dir, "xdg")
    os.makedirs(xdg_cache, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)

    hf_home = os.path.join(model_cache_dir, "hf")
    os.makedirs(hf_home, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))


def _load_semantic_model(model_name: str, device: str, *, offline: bool = False, model_cache: Optional[str] = None):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Semantic metrics require sentence-transformers. Install with: pip install sentence-transformers") from exc
    kwargs: dict[str, Any] = {"device": device}
    if model_cache:
        kwargs["cache_folder"] = os.path.join(model_cache, "hf")
    if offline:
        kwargs["local_files_only"] = True
    return SentenceTransformer(model_name, **kwargs)


def _semantic_similarity(model, hyp_text: str, ref_text: str) -> float:
    embeddings = model.encode([hyp_text, ref_text], normalize_embeddings=True)
    hyp_vec, ref_vec = embeddings[0], embeddings[1]
    return float((hyp_vec * ref_vec).sum())


def _compute_semantic_metrics(
    *,
    hyp_utterances: List[Utterance],
    ref_utterances: List[Utterance],
    base_wer: Optional[float],
    lang_code: Optional[str],
    semantic_model,
    saer_alpha: float,
) -> dict[str, Optional[float]]:
    hyp_text = _concat_text(hyp_utterances)
    ref_text = _concat_text(ref_utterances)
    if not hyp_text or not ref_text:
        return {"sem": None, "saer": None, "cer": None}

    sem = _semantic_similarity(semantic_model, hyp_text, ref_text)
    semantic_error = 1.0 - sem

    if _is_logographic(lang_code):
        cer_value = _cer(hyp_text, ref_text)
        form_error = cer_value
    else:
        cer_value = None
        form_error = base_wer

    if form_error is None:
        saer = None
    else:
        saer = (1.0 - saer_alpha) * semantic_error + saer_alpha * form_error

    return {"sem": sem, "saer": saer, "cer": cer_value}


def _inject_scores(output_json_path: Path, metric_values: dict[str, float]) -> None:
    try:
        if not output_json_path.exists():
            return
        with output_json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return
        score = payload.get("score", {}) if isinstance(payload.get("score"), dict) else {}
        score.update(
            {
                "wer": metric_values.get("wer"),
                "wder": metric_values.get("wder"),
                "cpwer": metric_values.get("cpwer"),
                "spkcntmae": metric_values.get("spkcntmae"),
                "cer": metric_values.get("cer"),
                "sem": metric_values.get("sem"),
                "saer": metric_values.get("saer"),
            }
        )
        payload["score"] = score
        with output_json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception:  # pylint: disable=broad-exception-caught
        LOG.debug("Failed to inject scores into %s", output_json_path, exc_info=True)


def _is_nonempty_json(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _cleanup_failed_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        try:
            if child.is_file() and child.stat().st_size == 0:
                child.unlink()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    try:
        if not any(output_dir.iterdir()):
            output_dir.rmdir()
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def run_item(
    args: argparse.Namespace,
    *,
    item: DatasetItem,
    system_name: str,
    log_level: int,
    models: Optional[Any] = None,
    semantic_model=None,
    saer_alpha: float = 0.5,
) -> RunResult:
    system_dir = Path(args.outdir).resolve() / system_name / item.item_id
    temp_dir = system_dir.with_name(f"{system_dir.name}__partial")
    work_dir = Path(args.workdir).resolve() / system_name / item.item_id if args.workdir else None
    temp_dir.mkdir(parents=True, exist_ok=True)
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)

    run_args = _build_run_namespace(args, item=item, system_name=system_name, item_output_dir=temp_dir, item_work_dir=work_dir)
    config = make_config(run_args)
    _apply_config_overrides(config, args)
    output_formats = build_output_formats(run_args, default_stdout=False)
    output_prefix_no_ext, working_prefix_no_ext = build_prefixes(config, str(item.audio_path))
    output_json_path = Path(f"{output_prefix_no_ext}.json")
    final_output_json = system_dir / output_json_path.name
    final_output_txt = final_output_json.with_suffix(".txt")

    existing_outputs_ok = _is_nonempty_json(final_output_json) and (not args.txt or _is_nonempty_file(final_output_txt))
    if args.skip_existing and existing_outputs_ok:
        hyp_utterances = read_utterances(str(final_output_json))
        scored_hyp_utterances = _filter_scored_utterances(hyp_utterances)
        scored_ref_utterances = _filter_scored_utterances(item.reference_utterances)
        lang_code = item.languages[0] if item.languages else None
        normalized_hyp_utterances = _normalize_form_utterances(scored_hyp_utterances, lang_code)
        if item.normalized_reference_text:
            normalized_ref_utterances = [
                Utterance(
                    utterance_id="utt1",
                    speaker=None,
                    start_ts=0,
                    end_ts=0,
                    text=item.normalized_reference_text,
                    words=[],
                )
            ]
        else:
            normalized_ref_utterances = _normalize_form_utterances(scored_ref_utterances, lang_code)
        metrics = compute_metrics(normalized_hyp_utterances, normalized_ref_utterances)
        metric_values = _metrics_to_result(metrics)
        count_values = _compute_corpus_counts(
            scored_hyp_utterances,
            scored_ref_utterances,
            lang_code,
            normalized_ref_text=item.normalized_reference_text,
        )
        metric_values.update(count_values)
        if semantic_model is not None:
            sem_metrics = _compute_semantic_metrics(
                hyp_utterances=scored_hyp_utterances,
                ref_utterances=scored_ref_utterances,
                base_wer=metric_values.get("wer"),
                lang_code=lang_code,
                semantic_model=semantic_model,
                saer_alpha=saer_alpha,
            )
            metric_values.update(sem_metrics)
        if args.txt:
            _write_score_text_artifacts(final_output_json, scored_hyp_utterances, lang_code)
        _inject_scores(final_output_json, metric_values)
        LOG.info(
            "Metrics %s/%s (cached): WER=%.4f WDER=%.4f cpWER=%.4f SpkCntMAE=%.4f",
            system_name,
            item.item_id,
            metric_values["wer"],
            metric_values["wder"],
            metric_values["cpwer"],
            metric_values["spkcntmae"],
        )
        if metric_values.get("saer") is not None:
            LOG.info(
                "Semantic %s/%s (cached): SEM=%.4f SAER=%.4f",
                system_name,
                item.item_id,
                metric_values["sem"],
                metric_values["saer"],
            )
        return RunResult(
            item_id=item.item_id,
            system=system_name,
            status="skipped_existing",
            audio_path=str(item.audio_path),
            output_json=str(final_output_json),
            elapsed_seconds=0.0,
            languages=item.languages,
            wer=metric_values["wer"],
            wder=metric_values["wder"],
            cpwer=metric_values["cpwer"],
            spkcntmae=metric_values["spkcntmae"],
            cer=metric_values.get("cer"),
            sem=metric_values.get("sem"),
            saer=metric_values.get("saer"),
            wer_sub=metric_values.get("wer_sub"),
            wer_del=metric_values.get("wer_del"),
            wer_ins=metric_values.get("wer_ins"),
            wer_total=metric_values.get("wer_total"),
            cer_dist=metric_values.get("cer_dist"),
            cer_total=metric_values.get("cer_total"),
            metadata=item.metadata,
        )

    start = perf_counter()
    try:
        mode = _system_mode(system_name)
        forced_language = item.languages[0] if (_fixed_primary_language(system_name) and item.languages) else None
        if mode == "pipeline":
            success = run_single_input(
                args=run_args,
                log_level=log_level,
                source_path=str(item.audio_path),
                config=config,
                models=models,
                output_prefix_no_ext=output_prefix_no_ext,
                working_prefix_no_ext=working_prefix_no_ext,
                output_formats=output_formats,
                default_stdout=False,
            )
            if not success:
                raise RuntimeError(f"Run rejected for {item.audio_path}")
            if not _is_nonempty_json(output_json_path):
                raise RuntimeError("Run completed without a valid JSON output.")
        elif mode == "whisper_baseline":
            if models is None:
                raise RuntimeError("Whisper baseline runner is not initialized.")
            output_json_path, _ = models.transcribe(item.audio_path, temp_dir, item.audio_path.stem, forced_language=forced_language)
        elif mode == "qwen_baseline":
            if models is None:
                raise RuntimeError("Qwen baseline runner is not initialized.")
            output_json_path, _ = models.transcribe(
                item.audio_path,
                temp_dir,
                item.audio_path.stem,
                item.languages or ["en"],
                forced_language=forced_language,
            )
        else:
            raise RuntimeError(f"Unsupported system mode: {mode}")
        if not _is_nonempty_json(output_json_path):
            raise RuntimeError("Run completed without a valid JSON output.")
        hyp_utterances = read_utterances(str(output_json_path))
        scored_hyp_utterances = _filter_scored_utterances(hyp_utterances)
        scored_ref_utterances = _filter_scored_utterances(item.reference_utterances)
        lang_code = item.languages[0] if item.languages else None
        normalized_hyp_utterances = _normalize_form_utterances(scored_hyp_utterances, lang_code)
        if item.normalized_reference_text:
            normalized_ref_utterances = [
                Utterance(
                    utterance_id="utt1",
                    speaker=None,
                    start_ts=0,
                    end_ts=0,
                    text=item.normalized_reference_text,
                    words=[],
                )
            ]
        else:
            normalized_ref_utterances = _normalize_form_utterances(scored_ref_utterances, lang_code)
        metrics = compute_metrics(normalized_hyp_utterances, normalized_ref_utterances)
        metric_values = _metrics_to_result(metrics)
        count_values = _compute_corpus_counts(
            scored_hyp_utterances,
            scored_ref_utterances,
            lang_code,
            normalized_ref_text=item.normalized_reference_text,
        )
        metric_values.update(count_values)
        if semantic_model is not None:
            sem_metrics = _compute_semantic_metrics(
                hyp_utterances=scored_hyp_utterances,
                ref_utterances=scored_ref_utterances,
                base_wer=metric_values.get("wer"),
                lang_code=lang_code,
                semantic_model=semantic_model,
                saer_alpha=saer_alpha,
            )
            metric_values.update(sem_metrics)
        if system_dir.exists():
            shutil.rmtree(system_dir, ignore_errors=True)
        shutil.move(str(temp_dir), str(system_dir))
        output_json_path = system_dir / output_json_path.name
        if args.txt:
            _write_score_text_artifacts(output_json_path, scored_hyp_utterances, lang_code)
        _inject_scores(output_json_path, metric_values)
        LOG.info(
            "Metrics %s/%s: WER=%.4f WDER=%.4f cpWER=%.4f SpkCntMAE=%.4f",
            system_name,
            item.item_id,
            metric_values["wer"],
            metric_values["wder"],
            metric_values["cpwer"],
            metric_values["spkcntmae"],
        )
        if metric_values.get("saer") is not None:
            LOG.info(
                "Semantic %s/%s: SEM=%.4f SAER=%.4f",
                system_name,
                item.item_id,
                metric_values["sem"],
                metric_values["saer"],
            )
        elapsed = perf_counter() - start
        return RunResult(
            item_id=item.item_id,
            system=system_name,
            status="ok",
            audio_path=str(item.audio_path),
            output_json=str(output_json_path),
            elapsed_seconds=elapsed,
            languages=item.languages,
            wer=metric_values["wer"],
            wder=metric_values["wder"],
            cpwer=metric_values["cpwer"],
            spkcntmae=metric_values["spkcntmae"],
            cer=metric_values.get("cer"),
            sem=metric_values.get("sem"),
            saer=metric_values.get("saer"),
            wer_sub=metric_values.get("wer_sub"),
            wer_del=metric_values.get("wer_del"),
            wer_ins=metric_values.get("wer_ins"),
            wer_total=metric_values.get("wer_total"),
            cer_dist=metric_values.get("cer_dist"),
            cer_total=metric_values.get("cer_total"),
            metadata=item.metadata,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        elapsed = perf_counter() - start
        _cleanup_failed_outputs(temp_dir)
        _cleanup_failed_outputs(system_dir)
        return RunResult(
            item_id=item.item_id,
            system=system_name,
            status="failed",
            audio_path=str(item.audio_path),
            output_json=str(output_json_path) if output_json_path.exists() else None,
            elapsed_seconds=elapsed,
            languages=item.languages,
            error=str(exc),
            metadata=item.metadata,
        )


def _safe_mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _safe_median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def summarize_results(results: List[RunResult]) -> dict[str, Any]:
    normalized_results = [result if isinstance(result, RunResult) else RunResult(**result) for result in results]
    systems: dict[str, Any] = {}
    for system_name in sorted({result.system for result in normalized_results}):
        system_results = [result for result in normalized_results if result.system == system_name]
        successful = [result for result in system_results if result.status in ("ok", "skipped_existing")]
        failed = [result for result in system_results if result.status == "failed"]

        wer_values = [result.wer for result in successful if result.wer is not None and not math.isnan(result.wer)]
        cpwer_values = [result.cpwer for result in successful if result.cpwer is not None and not math.isnan(result.cpwer)]
        cer_values = [result.cer for result in successful if result.cer is not None and not math.isnan(result.cer)]
        sem_values = [result.sem for result in successful if result.sem is not None and not math.isnan(result.sem)]
        saer_values = [result.saer for result in successful if result.saer is not None and not math.isnan(result.saer)]
        elapsed_values = [result.elapsed_seconds for result in successful]

        # Corpus-level (weighted) WER/CER are derived from total insert/delete/sub counts.
        # We don't have those counts here, so compute by re-reading the output jsons.
        corpus_wer = None
        corpus_cer = None
        total_words = 0
        total_chars = 0
        if successful:
            try:
                from verbatim.eval.diarization_metrics import compute_utterance_metrics
                from verbatim.eval.diarization_utils import normalize_text

                wer_sub = wer_del = wer_ins = wer_total = 0
                cer_dist = 0
                cer_total = 0
                for result in successful:
                    if not result.output_json:
                        continue
                    try:
                        data = json.loads(Path(result.output_json).read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    hyp_utts = data.get("utterances", [])
                    hyp_text = " ".join(
                        u.get("text", "").strip()
                        for u in hyp_utts
                        if u.get("text")
                        and not (
                            not u.get("words") and isinstance(u.get("text"), str) and NONLEXICAL_UTTERANCE_RE.fullmatch(u.get("text", "").strip())
                        )
                    )
                    ref_text = " ".join(u.text.strip() for u in result.metadata.get("reference_utterances", [])) if False else None
                    # Fall back to run-time reference utterances when available
                    ref_text = " ".join(u.text.strip() for u in result.metadata.get("ref_utterances", [])) if False else ref_text
                    if ref_text is None:
                        # Use cached reference from result if present (not stored today)
                        # This is best-effort; missing ref_text will skip corpus computation for that item.
                        continue
                    metrics = compute_utterance_metrics(hyp_text=hyp_text, ref_text=ref_text)
                    wer_sub += metrics.wer_sub
                    wer_del += metrics.wer_delete
                    wer_ins += metrics.wer_insert
                    wer_total += metrics.wer_total
                    ref_norm = normalize_text(ref_text)
                    hyp_norm = normalize_text(hyp_text)
                    if ref_norm:
                        a = list(hyp_norm)
                        b = list(ref_norm)
                        prev = list(range(len(b) + 1))
                        for i, ca in enumerate(a, start=1):
                            cur = [i]
                            for j, cb in enumerate(b, start=1):
                                cost = 0 if ca == cb else 1
                                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
                            prev = cur
                        cer_dist += prev[-1]
                        cer_total += len(b)
                if wer_total:
                    corpus_wer = (wer_sub + wer_del + wer_ins) / wer_total
                    total_words = wer_total
                if cer_total:
                    corpus_cer = cer_dist / cer_total
                    total_chars = cer_total
            except Exception:  # pylint: disable=broad-exception-caught
                corpus_wer = None
                corpus_cer = None

        systems[system_name] = {
            "count": len(system_results),
            "successful": len(successful),
            "failed": len(failed),
            "mean_wer": _safe_mean(wer_values),
            "median_wer": _safe_median(wer_values),
            "mean_cpwer": _safe_mean(cpwer_values),
            "mean_cer": _safe_mean(cer_values),
            "mean_sem": _safe_mean(sem_values),
            "mean_saer": _safe_mean(saer_values),
            "corpus_wer": corpus_wer,
            "corpus_cer": corpus_cer,
            "corpus_total_words": total_words,
            "corpus_total_chars": total_chars,
            "mean_elapsed_seconds": _safe_mean(elapsed_values),
        }

    return {
        "systems": systems,
        "results": [asdict(result) for result in normalized_results],
    }


def _write_summary(outdir: Path, summary: dict[str, Any]) -> None:
    summary_json = outdir / "summary.json"
    summary_csv = outdir / "summary.csv"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    rows = summary["results"]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "system",
            "item_id",
            "status",
            "audio_path",
            "output_json",
            "elapsed_seconds",
            "wer",
            "wder",
            "cpwer",
            "spkcntmae",
            "cer",
            "sem",
            "saer",
            "languages",
            "error",
            "topic",
            "conversation_type",
            "cs_type",
            "cs_ratio",
            "first_language",
            "second_language",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metadata = row.get("metadata", {})
            writer.writerow(
                {
                    "system": row["system"],
                    "item_id": row["item_id"],
                    "status": row["status"],
                    "audio_path": row["audio_path"],
                    "output_json": row["output_json"],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "wer": row["wer"],
                    "wder": row["wder"],
                    "cpwer": row["cpwer"],
                    "spkcntmae": row["spkcntmae"],
                    "cer": row.get("cer"),
                    "sem": row.get("sem"),
                    "saer": row.get("saer"),
                    "languages": ",".join(row.get("languages", [])),
                    "error": row["error"],
                    "topic": metadata.get("topic"),
                    "conversation_type": metadata.get("conversation_type"),
                    "cs_type": metadata.get("cs_type"),
                    "cs_ratio": metadata.get("cs_ratio"),
                    "first_language": metadata.get("first_language"),
                    "second_language": metadata.get("second_language"),
                }
            )


def _release_models(models: Optional[Models]) -> None:
    if models is None:
        return
    del models
    gc.collect()
    try:  # Optional torch cleanup
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def main() -> int:
    parser = build_switchlingua_parser()
    args = parser.parse_args()
    global DEFAULT_SYSTEMS  # pylint: disable=global-statement
    DEFAULT_SYSTEMS = load_system_specs(Path(args.systems_config))

    if args.list_systems:
        list_systems()
        return 0

    if args.install_switchlingua:
        _ensure_switchlingua_repo(install=True)

    unknown = [system for system in args.systems if system not in DEFAULT_SYSTEMS]
    if unknown:
        parser.error(f"Unknown systems: {', '.join(unknown)}")

    log_level = compute_log_level(args.verbose)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    load_env_file()
    _configure_model_cache_env(args.model_cache, args.offline)

    if args.lang:
        manifest_path, audio_root, run_name = _resolve_lang_defaults(args.lang)
        if args.manifest is None:
            args.manifest = str(manifest_path)
        if args.audio_root is None:
            args.audio_root = str(audio_root)
        if args.run_name is None:
            args.run_name = run_name
        if not args.skip_existing:
            args.skip_existing = True
        if not args.saer:
            args.saer = True
        if args.saer_device is None:
            args.saer_device = "cuda" if not args.cpu else "cpu"
        if not args.txt:
            args.txt = True

    if args.manifest is None:
        parser.error("--manifest is required unless --lang is provided.")

    outdir = Path(args.outdir).resolve()
    if args.run_name:
        outdir = outdir / args.run_name
    # Ensure downstream uses the run-scoped output directory.
    args.outdir = str(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve() if args.audio_root else None
    reference_root = Path(args.reference_root).expanduser().resolve() if args.reference_root else None
    default_languages = _default_languages_for_lang(args.lang)
    items = load_manifest(
        manifest_path,
        audio_root=audio_root,
        reference_root=reference_root,
        limit=args.limit,
        default_languages=default_languages,
        lang_filter=args.lang,
    )

    LOG.info("Loaded %d dataset items from %s", len(items), manifest_path)
    results: List[RunResult] = []
    semantic_model = None
    if args.saer:
        saer_device = args.saer_device or ("cuda" if not args.cpu else "cpu")
        LOG.info("Loading semantic model %s on %s", args.saer_model, saer_device)
        semantic_model = _load_semantic_model(
            args.saer_model,
            saer_device,
            offline=args.offline,
            model_cache=args.model_cache,
        )

    for system_name in args.systems:
        LOG.info("Running system %s over %d items", system_name, len(items))
        shared_models: Optional[Any] = None
        for index, item in enumerate(items):
            LOG.info("Running %s on %s", system_name, item.audio_path)
            if shared_models is None:
                if _system_mode(system_name) == "pipeline":
                    run_args = _build_run_namespace(
                        args,
                        item=item,
                        system_name=system_name,
                        item_output_dir=Path(args.outdir).resolve() / system_name / item.item_id,
                        item_work_dir=Path(args.workdir).resolve() / system_name / item.item_id if args.workdir else None,
                    )
                    config = make_config(run_args)
                    _apply_config_overrides(config, args)
                    shared_models = Models(
                        device=config.device,
                        whisper_model_size=config.whisper_model_size,
                        voxtral_model_size=config.voxtral_model_size,
                        voxtral_dtype=config.voxtral_dtype,
                        voxtral_max_new_tokens=config.voxtral_max_new_tokens,
                        stream=config.stream,
                        transcriber_backend=config.transcriber_backend,
                        qwen_asr_model_size=config.qwen_asr_model_size,
                        qwen_aligner_model_size=config.qwen_aligner_model_size,
                        qwen_dtype=config.qwen_dtype,
                        qwen_max_inference_batch_size=config.qwen_max_inference_batch_size,
                        qwen_max_new_tokens=config.qwen_max_new_tokens,
                    )
                else:
                    shared_models = _create_shared_runner(args, system_name)
            result = run_item(
                args,
                item=item,
                system_name=system_name,
                log_level=log_level,
                models=shared_models,
                semantic_model=semantic_model,
                saer_alpha=args.saer_alpha,
            )
            results.append(result)
            if result.status == "failed":
                LOG.error("Run failed for %s / %s: %s", system_name, item.item_id, result.error)
                if not args.continue_on_error:
                    _write_summary(outdir, summarize_results(results))
                    _release_models(shared_models)
                    return 1
        _release_models(shared_models)

    summary = summarize_results(results)
    _write_summary(outdir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
