import logging
import os
import sys
from typing import Optional

LOG = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def apply_cache_env(model_cache_dir: Optional[str], offline: bool = False) -> None:
    """Apply cache/offline environment variables for this process.

    Mirrors Config.configure_cache without importing the full Config to avoid side effects.
    """
    if offline:
        os.environ["VERBATIM_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.setdefault("VERBATIM_OFFLINE", "0")

    # Default to local project cache if unspecified
    if not model_cache_dir:
        model_cache_dir = os.path.join(os.getcwd(), ".verbatim")

    if model_cache_dir:
        os.makedirs(model_cache_dir, exist_ok=True)
        os.environ["VERBATIM_MODEL_CACHE"] = model_cache_dir

        xdg_cache = os.path.join(model_cache_dir, "xdg")
        os.makedirs(xdg_cache, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", xdg_cache)

        whisper_cache = os.path.join(model_cache_dir, "whisper")
        os.makedirs(whisper_cache, exist_ok=True)
        os.environ.setdefault("WHISPER_CACHE_DIR", whisper_cache)

        hf_home = os.path.join(model_cache_dir, "hf")
        os.makedirs(hf_home, exist_ok=True)
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))


def prefetch(
    *,
    model_cache_dir: Optional[str],
    whisper_size: str = "large-v3",
    include_pyannote: bool = True,
    include_whisper_openai: bool = True,
    include_faster_whisper: bool = True,
    include_mlx_whisper: bool = sys.platform == "darwin" and os.uname().machine in ("arm64", "aarch64"),
) -> None:
    """Prefetch commonly used models into the cache.

    This function attempts to only download/copy into the provided cache directory,
    so later runs can use --offline.
    """
    apply_cache_env(model_cache_dir, offline=False)

    # 1) Hugging Face models via huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover - optional path
        LOG.warning(f"huggingface_hub not available: {e}")
        snapshot_download = None  # type: ignore

    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Pyannote diarization/separation
    if include_pyannote and snapshot_download is not None:
        for repo in ("pyannote/speaker-diarization-3.1", "pyannote/speech-separation-ami-1.0"):
            try:
                LOG.info(f"Prefetching HF repo: {repo}")
                snapshot_download(repo_id=repo, token=hf_token, local_files_only=False)
            except Exception as e:  # pragma: no cover
                LOG.warning(f"Failed to prefetch {repo}: {e}")

    # MLX Whisper models (macOS/Apple Silicon) hosted on HF
    if include_mlx_whisper and snapshot_download is not None:
        mlx_repo = f"mlx-community/whisper-{whisper_size}-mlx"
        try:
            LOG.info(f"Prefetching HF repo: {mlx_repo}")
            snapshot_download(repo_id=mlx_repo, token=hf_token, local_files_only=False)
        except Exception as e:  # pragma: no cover
            LOG.warning(f"Failed to prefetch {mlx_repo}: {e}")

    # 2) Faster-Whisper (uses internal download to a local root)
    if include_faster_whisper:
        try:
            from faster_whisper import WhisperModel
            cache_root = os.getenv("VERBATIM_MODEL_CACHE")
            download_root = os.path.join(cache_root, "faster-whisper") if cache_root else None
            LOG.info(f"Prefetching faster-whisper model: {whisper_size}")
            try:
                model = WhisperModel(
                    model_size_or_path=whisper_size,
                    device="cpu",
                    compute_type="default",
                    download_root=download_root,
                    local_files_only=False,
                )
            except TypeError:
                model = WhisperModel(
                    model_size_or_path=whisper_size,
                    device="cpu",
                    compute_type="default",
                )
            # Touch a trivial call to ensure weights are resolved
            _ = model.transcribe(audio=[0.0], language="en")[0]
            del model
        except Exception as e:  # pragma: no cover
            LOG.warning(f"Failed to prefetch faster-whisper {whisper_size}: {e}")

    # 3) OpenAI Whisper (downloads .pt to WHISPER_CACHE_DIR)
    if include_whisper_openai:
        try:
            import whisper as openai_whisper

            LOG.info(f"Prefetching OpenAI whisper model: {whisper_size}")
            model = openai_whisper.load_model(whisper_size, device="cpu")
            del model
        except Exception as e:  # pragma: no cover
            LOG.warning(f"Failed to prefetch openai/whisper {whisper_size}: {e}")

    # 4) Voice isolation: provide guidance; download may be manual depending on backend
    cache_root = os.getenv("VERBATIM_MODEL_CACHE")
    if cache_root:
        iso_dir = os.path.join(cache_root, "audio-separator")
        os.makedirs(iso_dir, exist_ok=True)
        LOG.info(
            f"If needed, place MDX checkpoint (e.g., MDX23C-8KFFT-InstVoc_HQ_2.ckpt) under: {iso_dir} to enable offline isolation"
        )
