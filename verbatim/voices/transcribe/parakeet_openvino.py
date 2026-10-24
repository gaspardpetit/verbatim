"""Parakeet TDT v3 on Intel NPU through onnx-asr and OpenVINO."""

import hashlib
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from verbatim.model_cache import get_model_cache_dir, is_local_model_reference, prefetch_hf_snapshot, resolve_hf_snapshot_path
from verbatim_audio.settings import AUDIO_PARAMS
from verbatim_transcript import TranscriberCapabilities

from ...transcript.words import Word
from .transcribe import Transcriber

LOG = logging.getLogger(__name__)
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
MODEL_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
SAMPLE_RATE = 16000
BUCKET_FRAMES = (500, 1000, 2000, 3200)
FINAL_TOKEN_TAIL_SECONDS = 0.5


def _model_dir(reference: str) -> str:
    if reference in ("nvidia/parakeet-tdt-0.6b-v3", MODEL_NAME):
        reference = MODEL_REPO
    if is_local_model_reference(reference):
        return os.path.expanduser(reference)
    return resolve_hf_snapshot_path(reference, purpose="Parakeet OpenVINO model", offline=True)


def _compiled_dir(model_dir: Path, openvino_version: str) -> Path:
    root = get_model_cache_dir() or os.path.join(os.getcwd(), ".verbatim")
    files = ("encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx")
    # Content identifies the model even when the prepared cache moves to another machine.
    identity = hashlib.sha256(openvino_version.encode("utf-8"))
    for name in files:
        identity.update(name.encode("utf-8"))
        with (model_dir / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                identity.update(chunk)
    digest = identity.hexdigest()[:16]
    return Path(root) / "parakeet-openvino" / digest


def _blob_path(compiled_dir: Path, frames: Optional[int] = None) -> Path:
    return compiled_dir / ("decoder.blob" if frames is None else f"encoder-{frames}.blob")


def _save_compiled_model(compiled: Any, path: Path) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            exported = compiled.export_model()
            if isinstance(exported, io.BytesIO):
                exported.seek(0)
                shutil.copyfileobj(exported, stream)
            else:
                stream.write(exported)
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def prefetch_parakeet_openvino_model(reference: str) -> None:
    """Download and compile every supported NPU shape during setup."""
    if reference in ("nvidia/parakeet-tdt-0.6b-v3", MODEL_NAME):
        reference = MODEL_REPO
    model_dir = Path(os.path.expanduser(reference)) if is_local_model_reference(reference) else Path(prefetch_hf_snapshot(reference))
    try:
        import openvino as ov  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("Parakeet NPU requires `verbatim[parakeet_openvino]`.") from exc
    core = ov.Core()
    if "NPU" not in core.available_devices:
        raise RuntimeError(f"Intel NPU is unavailable to OpenVINO; available devices: {core.available_devices}")
    compiled_dir = _compiled_dir(model_dir, ov.__version__)
    compiled_dir.mkdir(parents=True, exist_ok=True)

    decoder_path = _blob_path(compiled_dir)
    if not decoder_path.is_file():
        decoder = core.read_model(str(model_dir / "decoder_joint-model.onnx"))
        decoder.reshape({
            "encoder_outputs": [1, 1024, 1], "targets": [1, 1], "target_length": [1],
            "input_states_1": [2, 1, 640], "input_states_2": [2, 1, 640],
        })
        _save_compiled_model(core.compile_model(decoder, "NPU", {"PERFORMANCE_HINT": "LATENCY"}), decoder_path)

    encoder_path = model_dir / "encoder-model.onnx"
    for frames in BUCKET_FRAMES:
        blob_path = _blob_path(compiled_dir, frames)
        if blob_path.is_file():
            continue
        LOG.info("Preparing Parakeet NPU encoder for %.1f seconds", frames / 100)
        encoder = core.read_model(str(encoder_path))
        encoder.reshape({"audio_signal": [1, 128, frames], "length": [1]})
        _save_compiled_model(core.compile_model(encoder, "NPU", {"PERFORMANCE_HINT": "LATENCY"}), blob_path)


class _Port:
    def __init__(self, port: Any):
        self.name = port.any_name
        self.shape = list(port.shape)


class _OpenVinoSession:
    """The small part of the ONNX Runtime session API used by onnx-asr."""

    def __init__(self, compiled: Any):
        self._compiled = compiled
        self._request = self._compiled.create_infer_request()

    def get_inputs(self) -> list[_Port]:
        return [_Port(port) for port in self._compiled.inputs]

    def get_outputs(self) -> list[_Port]:
        return [_Port(port) for port in self._compiled.outputs]

    def run(self, output_names: list[str], feed: dict[str, Any]) -> list[NDArray]:
        typed_feed = {
            name: np.asarray(value, dtype=np.int32 if name in ("targets", "target_length") else None)
            for name, value in feed.items()
        }
        values = self._request.infer(typed_feed)
        by_name = {port.any_name: np.asarray(value) for port, value in values.items()}
        state_outputs = sorted((name, value) for name, value in by_name.items() if value.ndim == 3 and value.shape[0] == 2)
        resolved = []
        for name in output_names:
            if name in by_name:
                resolved.append(by_name[name])
            elif name == "encoded_lengths":
                resolved.append(np.asarray(next(value for value in by_name.values() if value.ndim == 1), dtype=np.int64))
            elif name == "outputs":
                rank = 4 if "targets" in feed else 3
                resolved.append(next(value for value in by_name.values() if value.ndim == rank))
            elif name in ("output_states_1", "output_states_2"):
                resolved.append(state_outputs[int(name[-1]) - 1][1])
            else:
                raise KeyError(f"OpenVINO model has no output {name}")
        return resolved


class _BucketedEncoder:
    def __init__(self, core: Any, compiled_dir: Path):
        self._core = core
        self._compiled_dir = compiled_dir
        self._sessions: dict[int, _OpenVinoSession] = {}
        self._sessions[BUCKET_FRAMES[0]] = self._load(BUCKET_FRAMES[0])

    def _load(self, frames: int) -> _OpenVinoSession:
        LOG.info("Loading prepared Parakeet NPU encoder for %.1f seconds", frames / 100)
        return _OpenVinoSession(self._core.import_model(_blob_path(self._compiled_dir, frames).read_bytes(), "NPU"))

    def get_inputs(self) -> list[_Port]:
        return self._sessions[BUCKET_FRAMES[0]].get_inputs()

    def get_outputs(self) -> list[_Port]:
        return self._sessions[BUCKET_FRAMES[0]].get_outputs()

    def run(self, output_names: list[str], feed: dict[str, NDArray]) -> list[NDArray]:
        audio = feed["audio_signal"]
        frames = audio.shape[-1]
        bucket = next((size for size in BUCKET_FRAMES if frames <= size), None)
        if bucket is None:
            raise ValueError(f"Parakeet NPU window exceeds {BUCKET_FRAMES[-1] / 100:g} seconds")
        if bucket not in self._sessions:
            self._sessions[bucket] = self._load(bucket)
        padded = np.zeros((1, 128, bucket), dtype=np.float32)
        padded[:, :, :frames] = audio
        return self._sessions[bucket].run(output_names, {"audio_signal": padded, "length": feed["length"]})


def _word_results(result: Any, window_ts: int, audio_ts: int) -> List[Word]:
    tokens = result.tokens or []
    stamps = result.timestamps or []
    if len(tokens) != len(stamps):
        raise RuntimeError("Parakeet ONNX returned tokens without matching timestamps")
    words: List[Word] = []
    current = ""
    start = 0.0
    for index, (token, stamp) in enumerate(zip(tokens, stamps)):
        if not words and not current:
            token = token.lstrip()
        if token.startswith(" ") and current.strip():
            end = float(stamp)
            first = max(window_ts, min(window_ts + round(start * SAMPLE_RATE), audio_ts))
            last = max(first, min(window_ts + round(end * SAMPLE_RATE), audio_ts))
            # TDT can emit several words at one encoder timestep.
            last = min(audio_ts, max(last, first + 1))
            words.append(Word(start_ts=first, end_ts=last, word=current, probability=1.0, lang="und"))
            current = ""
        if not current:
            start = float(stamp)
        current += token
        if index == len(tokens) - 1 and current.strip():
            first = max(window_ts, min(window_ts + round(start * SAMPLE_RATE), audio_ts))
            # Token timestamps are onsets; the final token has no following onset to use as its end.
            last = max(first, min(window_ts + round((float(stamp) + FINAL_TOKEN_TAIL_SECONDS) * SAMPLE_RATE), audio_ts))
            last = min(audio_ts, max(last, first + 1))
            words.append(Word(start_ts=first, end_ts=last, word=current, probability=1.0, lang="und"))
    return words


class ParakeetOpenVinoTranscriber(Transcriber):
    """Run the ONNX Parakeet export with its encoder and decoder on Intel NPU."""

    CAPABILITIES = TranscriberCapabilities(
        accepts_language_hint=False,
        detects_language=False,
        reports_language=False,
        supports_prompt=False,
        supports_prefix=False,
        window_duration=30.0,
    )

    def __init__(self, *, model_size_or_path: str, device: str):
        _ = device  # Other pipeline stages retain their normal CPU/CUDA device.
        try:
            import onnx_asr  # pylint: disable=import-outside-toplevel
            import openvino as ov  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("Parakeet NPU requires `verbatim[parakeet_openvino]`.") from exc

        core = ov.Core()
        if "NPU" not in core.available_devices:
            raise RuntimeError(f"Intel NPU is unavailable to OpenVINO; available devices: {core.available_devices}")
        model_dir = Path(_model_dir(model_size_or_path))
        encoder = model_dir / "encoder-model.onnx"
        decoder = model_dir / "decoder_joint-model.onnx"
        for path in (encoder, model_dir / "encoder-model.onnx.data", decoder):
            if not path.is_file():
                raise RuntimeError(f"Parakeet NPU model is missing {path}; run --install first")

        compiled_dir = _compiled_dir(model_dir, ov.__version__)
        prepared_files = [_blob_path(compiled_dir), *(_blob_path(compiled_dir, frames) for frames in BUCKET_FRAMES)]
        missing = [path for path in prepared_files if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Parakeet NPU is not prepared; run `verbatim --asr-backend parakeet-openvino --install` "
                "with the same --modeldir before processing."
            )
        self._model: Any = onnx_asr.load_model(MODEL_NAME, path=str(model_dir), quantization="int8", providers=["CPUExecutionProvider"])
        self._model.asr._encoder = _BucketedEncoder(core, compiled_dir)  # pylint: disable=protected-access
        decoder_blob = _blob_path(compiled_dir).read_bytes()
        self._model.asr._decoder_joint = _OpenVinoSession(core.import_model(decoder_blob, "NPU"))  # pylint: disable=protected-access
        self._model = self._model.with_timestamps()

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        _ = audio, lang
        raise RuntimeError("Parakeet does not expose language identification; select a separate language identifier.")

    def transcribe(
        self, *, audio: NDArray, lang: Optional[str], prompt: str, prefix: str, window_ts: int, audio_ts: int,
        whisper_beam_size: int = 3, whisper_best_of: int = 3, whisper_patience: float = 1.0,
        whisper_temperatures: Optional[List[float]] = None,
    ) -> List[Word]:
        _ = prompt, prefix, whisper_beam_size, whisper_best_of, whisper_patience, whisper_temperatures
        if AUDIO_PARAMS.sample_rate != SAMPLE_RATE:
            raise RuntimeError(f"Parakeet requires a {SAMPLE_RATE} Hz Verbatim sample rate")
        if lang is not None:
            raise RuntimeError("Parakeet does not accept language hints")
        if audio.ndim != 1:
            raise ValueError("Parakeet requires mono audio")
        if not audio.size:
            return []
        words: List[Word] = []
        window_samples = 30 * SAMPLE_RATE
        for offset in range(0, audio.size, window_samples):
            chunk = np.ascontiguousarray(audio[offset : offset + window_samples], dtype=np.float32)
            chunk_ts = window_ts + offset
            chunk_end_ts = min(audio_ts, chunk_ts + chunk.size)
            result = self._model.recognize(chunk, sample_rate=SAMPLE_RATE)
            words.extend(_word_results(result, chunk_ts, chunk_end_ts))
        return words
