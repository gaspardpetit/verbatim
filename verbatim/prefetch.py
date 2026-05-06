import logging
import os
import platform
import re
import sys
from dataclasses import dataclass
from typing import Optional

from verbatim.config import Config
from verbatim_audio.sources.sourceconfig import SourceConfig

LOG = logging.getLogger(__name__)

DEFAULT_SENTENCE_TOKENIZER_MODEL = "sat-3l-sm"
DEFAULT_SENTENCE_TOKENIZER_TOKENIZER = "facebookAI/xlm-roberta-base"


@dataclass
class InstallRequirements:
    include_sat: bool = False
    include_mms_language_model: bool = False
    include_ast_noise_model: bool = False
    include_faster_whisper: bool = False
    include_mlx_whisper: bool = False
    include_qwen_asr: bool = False
    include_voxtral: bool = False
    include_pyannote_diarization: bool = False
    include_pyannote_separation: bool = False
    include_isolation: bool = False


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def apply_cache_env(model_cache_dir: Optional[str], offline: bool = False) -> None:
    """Apply cache/offline environment variables for this process."""
    if offline:
        os.environ["VERBATIM_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ["VERBATIM_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    if not model_cache_dir:
        model_cache_dir = os.path.join(os.getcwd(), ".verbatim")

    if model_cache_dir:
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ["VERBATIM_MODELDIR"] = model_cache_dir

        xdg_cache = os.path.join(model_cache_dir, "xdg")
        os.makedirs(xdg_cache, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = xdg_cache

        hf_home = os.path.join(model_cache_dir, "hf")
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = hf_home
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")

        pyannote_cache = os.path.join(model_cache_dir, "pyannote")
        os.makedirs(pyannote_cache, exist_ok=True)
        os.environ["PYANNOTE_CACHE"] = pyannote_cache
    LOG.debug(
        "Cache env: VERBATIM_MODELDIR=%s HF_HOME=%s HUGGINGFACE_HUB_CACHE=%s PYANNOTE_CACHE=%s",
        os.getenv("VERBATIM_MODELDIR"),
        os.getenv("HF_HOME"),
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.getenv("PYANNOTE_CACHE"),
    )


def _hf_models_dir(repo_id: str) -> str:
    parts = repo_id.split("/")
    if len(parts) != 2:
        safe = repo_id.replace("/", "--")
        return f"models--{safe}"
    return f"models--{parts[0]}--{parts[1]}"


def _classify_whisper_model(value: str) -> str:
    model_kind = "size"
    if not value:
        model_kind = "unknown"
    elif os.path.isabs(value):
        model_kind = "path"
    elif os.path.exists(value):
        model_kind = "path"
    elif value.startswith((".", "~")):
        model_kind = "path"
    elif ":" in value:
        model_kind = "path"
    elif os.path.altsep and os.path.altsep in value:
        model_kind = "path"
    elif "\\" in value:
        model_kind = "path"
    elif "/" in value:
        model_kind = "hf_repo" if value.count("/") == 1 else "path"
    return model_kind


def _resolve_hf_revision(repo_id: str, *, local_dir: Optional[str] = None) -> str:
    sha_re = re.compile(r"^[0-9a-f]{40}$")
    candidates = []
    models_dirname = _hf_models_dir(repo_id)

    if local_dir:
        candidates.append(os.path.join(local_dir, models_dirname, "refs", "main"))

    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if not hub_cache:
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            hub_cache = os.path.join(hf_home, "hub")
    if hub_cache:
        candidates.append(os.path.join(hub_cache, models_dirname, "refs", "main"))

    for ref_path in candidates:
        try:
            with open(ref_path, "r", encoding="utf-8") as handle:
                ref = handle.read().strip()
            if sha_re.match(ref):
                return ref
        except OSError:
            continue

    return "main"


def _parse_diarization_requirements(source_config: SourceConfig) -> tuple[bool, bool]:
    strategy = source_config.diarize_strategy
    if not strategy:
        return False, False

    if strategy in ("pyannote", "separate"):
        return strategy == "pyannote", strategy == "separate"

    if "=" not in strategy and ";" not in strategy:
        return False, False

    try:
        from verbatim_diarization.policy import parse_policy  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False, False

    include_pyannote = False
    include_separate = False
    for clause in parse_policy(strategy):
        include_pyannote = include_pyannote or clause.strategy == "pyannote"
        include_separate = include_separate or clause.strategy == "separate"
    return include_pyannote, include_separate


def collect_install_requirements(*, config: Config, source_config: SourceConfig) -> InstallRequirements:
    requirements = InstallRequirements()
    asr_backend = (config.transcriber_backend or "auto").lower()

    if not config.stream:
        requirements.include_sat = True

    if asr_backend in ("qwen", "qwen-asr"):
        requirements.include_qwen_asr = True
    elif asr_backend == "voxtral":
        requirements.include_voxtral = True
    else:
        requirements.include_faster_whisper = True
        requirements.include_mlx_whisper = sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")

    language_backend = (config.language_identifier_backend or "transcriber").lower()
    if language_backend == "mms" or asr_backend == "voxtral":
        requirements.include_mms_language_model = True

    if (config.non_speech_backend or "energy").lower() == "ast":
        requirements.include_ast_noise_model = True

    include_pyannote, include_separate = _parse_diarization_requirements(source_config)
    requirements.include_pyannote_diarization = include_pyannote or include_separate
    requirements.include_pyannote_separation = include_separate
    requirements.include_isolation = source_config.isolate is not None
    return requirements


def prefetch_faster_whisper_model(*, whisper_size: str, include_mlx_whisper: bool) -> None:
    try:  # type: ignore
        from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
        from huggingface_hub.errors import HfHubHTTPError  # pylint: disable=import-outside-toplevel
        from huggingface_hub.utils import LocalEntryNotFoundError  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        LOG.warning("huggingface_hub not available: cannot prefetch HF-hosted models")
        return

    model_kind = _classify_whisper_model(whisper_size)
    if model_kind == "path":
        LOG.info("Skipping whisper model prefetch for local/custom path: %s", whisper_size)
        return

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")

    if include_mlx_whisper:
        mlx_repo = f"mlx-community/whisper-{whisper_size}-mlx"
        mlx_rev = _resolve_hf_revision(mlx_repo)
        try:
            LOG.info("Prefetching HF repo: %s (rev=%s)", mlx_repo, mlx_rev)
            local_path = snapshot_download(  # nosec B615 - revision provided via variable
                repo_id=mlx_repo,
                token=hf_token,
                local_files_only=False,
                revision=mlx_rev,
                cache_dir=hub_cache,
            )
            LOG.info("Downloaded: %s (rev=%s) to %s", mlx_repo, mlx_rev, local_path)
        except HfHubHTTPError as exc:  # pragma: no cover
            LOG.warning("Failed to prefetch %s: %s", mlx_repo, exc)

    try:
        cache_root = os.getenv("VERBATIM_MODELDIR")
        download_root = os.path.join(cache_root, "faster-whisper") if cache_root else None
        fw_repo = f"Systran/faster-whisper-{whisper_size}" if model_kind != "hf_repo" else whisper_size
        fw_rev = _resolve_hf_revision(fw_repo, local_dir=download_root)

        if download_root:
            os.makedirs(download_root, exist_ok=True)
            try:
                snapshot_download(  # nosec B615 - revision provided via variable
                    repo_id=fw_repo,
                    local_files_only=True,
                    revision=fw_rev,
                    cache_dir=download_root,
                )
                LOG.info("Already cached: faster-whisper (%s)", whisper_size)
            except LocalEntryNotFoundError:
                LOG.info("Prefetching faster-whisper model: %s", whisper_size)
                local_path = snapshot_download(  # nosec B615 - revision provided via variable
                    repo_id=fw_repo,
                    local_files_only=False,
                    revision=fw_rev,
                    cache_dir=download_root,
                )
                LOG.info("Downloaded: %s (rev=%s) to %s", fw_repo, fw_rev, local_path)
        else:
            LOG.info("Prefetching faster-whisper model: %s", whisper_size)
            local_path = snapshot_download(  # nosec B615 - revision provided via variable
                repo_id=fw_repo,
                local_files_only=False,
                revision=fw_rev,
                cache_dir=hub_cache,
            )
            LOG.info("Downloaded (HF cache): %s (rev=%s) to %s", fw_repo, fw_rev, local_path)
    except (OSError, HfHubHTTPError) as exc:  # pragma: no cover
        LOG.warning("Failed to prefetch faster-whisper %s: %s", whisper_size, exc)


def prefetch(*, config: Config, source_config: SourceConfig) -> None:
    """Prefetch the models required by the effective runtime configuration."""
    apply_cache_env(config.model_cache_dir, offline=False)
    requirements = collect_install_requirements(config=config, source_config=source_config)

    if requirements.include_sat:
        from verbatim.transcript.sentences import prefetch_sentence_tokenizer_models  # pylint: disable=import-outside-toplevel

        prefetch_sentence_tokenizer_models(
            model_name_or_path=DEFAULT_SENTENCE_TOKENIZER_MODEL,
            tokenizer_name_or_path=DEFAULT_SENTENCE_TOKENIZER_TOKENIZER,
        )

    if requirements.include_mms_language_model:
        from verbatim.language_id import prefetch_language_identifier_models  # pylint: disable=import-outside-toplevel

        prefetch_language_identifier_models(config.mms_lid_model_size)

    if requirements.include_ast_noise_model:
        from verbatim.non_speech import prefetch_non_speech_models  # pylint: disable=import-outside-toplevel

        prefetch_non_speech_models(config.ast_audio_model_size)

    if requirements.include_qwen_asr:
        from verbatim.voices.transcribe.qwen_asr import prefetch_qwen_models  # pylint: disable=import-outside-toplevel

        prefetch_qwen_models(
            model_size_or_path=config.qwen_asr_model_size,
            aligner_model_size_or_path=config.qwen_aligner_model_size,
        )

    if requirements.include_voxtral:
        from verbatim.voices.transcribe.voxtral import prefetch_voxtral_models  # pylint: disable=import-outside-toplevel

        prefetch_voxtral_models(
            model_size_or_path=config.voxtral_model_size,
            aligner_model_size_or_path=config.qwen_aligner_model_size,
        )

    if requirements.include_faster_whisper:
        prefetch_faster_whisper_model(
            whisper_size=config.whisper_model_size,
            include_mlx_whisper=requirements.include_mlx_whisper,
        )

    if requirements.include_pyannote_diarization or requirements.include_pyannote_separation:
        try:
            from verbatim_diarization.prefetch import prefetch_diarization_models  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - defensive
            LOG.warning("verbatim_diarization not available; skipping diarization prefetch: %s", exc)
        else:
            prefetch_diarization_models(
                hf_token=os.getenv("HUGGINGFACE_TOKEN"),
                cache_dir=os.getenv("PYANNOTE_CACHE"),
                offline=False,
                include_separation=requirements.include_pyannote_separation,
            )

    if requirements.include_isolation:
        from verbatim.voices.isolation import prefetch_isolation_model  # pylint: disable=import-outside-toplevel

        prefetch_isolation_model()
