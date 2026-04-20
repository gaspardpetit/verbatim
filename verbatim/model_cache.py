import logging
import os
from typing import Any, Optional

LOG = logging.getLogger(__name__)


def is_offline_mode() -> bool:
    return os.getenv("VERBATIM_OFFLINE", "0").lower() in ("1", "true", "yes", "on")


def get_model_cache_dir() -> Optional[str]:
    return os.getenv("VERBATIM_MODELDIR")


def get_hf_cache_dir() -> Optional[str]:
    return os.getenv("HUGGINGFACE_HUB_CACHE")


def get_pyannote_cache_dir() -> Optional[str]:
    return os.getenv("PYANNOTE_CACHE")


def get_audio_separator_cache_dir() -> Optional[str]:
    cache_root = get_model_cache_dir()
    if not cache_root:
        return None
    return os.path.join(cache_root, "audio-separator")


def ensure_local_path(model_name_or_path: str) -> str:
    return os.path.expanduser(model_name_or_path)


def is_local_model_reference(model_name_or_path: str) -> bool:  # pylint: disable=too-many-return-statements
    candidate = ensure_local_path(model_name_or_path)
    if os.path.isabs(candidate):
        return True
    if os.path.exists(candidate):
        return True
    if candidate.startswith((".", "~")):
        return True
    if ":" in candidate:
        return True
    if "\\" in candidate:
        return True
    if candidate.startswith("/"):
        return True
    if candidate.count("/") > 1:
        return True
    return False


def build_transformers_load_kwargs(*, offline: Optional[bool] = None, cache_dir: Optional[str] = None) -> dict[str, Any]:
    if offline is None:
        offline = is_offline_mode()
    kwargs: dict[str, Any] = {
        "local_files_only": offline,
    }
    effective_cache_dir = cache_dir or get_hf_cache_dir()
    if effective_cache_dir:
        kwargs["cache_dir"] = effective_cache_dir
    return kwargs


def _snapshot_download(
    repo_id: str,
    *,
    cache_dir: Optional[str],
    local_files_only: bool,
    revision: str = "main",
    token: Optional[str] = None,
) -> str:
    try:
        from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to resolve cached model snapshots.") from exc

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_files_only": local_files_only,
        "revision": revision,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if token:
        kwargs["token"] = token
    return snapshot_download(  # nosec B615 - revision is provided explicitly by the caller
        repo_id=kwargs["repo_id"],
        local_files_only=kwargs["local_files_only"],
        revision=kwargs["revision"],
        cache_dir=kwargs.get("cache_dir"),
        token=kwargs.get("token"),
    )


def _hf_hub_download(
    repo_id: str,
    *,
    filename: str,
    cache_dir: Optional[str],
    local_files_only: bool,
    revision: str = "main",
    token: Optional[str] = None,
) -> str:
    try:
        from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("huggingface_hub is required to resolve cached model files.") from exc

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "local_files_only": local_files_only,
        "revision": revision,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if token:
        kwargs["token"] = token
        kwargs["use_auth_token"] = token
    return hf_hub_download(  # nosec B615 - revision is provided explicitly by the caller
        repo_id=kwargs["repo_id"],
        filename=kwargs["filename"],
        local_files_only=kwargs["local_files_only"],
        revision=kwargs["revision"],
        cache_dir=kwargs.get("cache_dir"),
        token=kwargs.get("token"),
        use_auth_token=kwargs.get("use_auth_token"),
    )


def resolve_hf_snapshot_path(
    model_name_or_path: str,
    *,
    purpose: str,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    token: Optional[str] = None,
    offline: Optional[bool] = None,
) -> str:
    candidate = ensure_local_path(model_name_or_path)
    if is_local_model_reference(candidate):
        return candidate

    effective_offline = is_offline_mode() if offline is None else offline
    effective_cache_dir = cache_dir or get_hf_cache_dir()
    if not effective_offline:
        return model_name_or_path

    try:
        return _snapshot_download(
            model_name_or_path,
            cache_dir=effective_cache_dir,
            local_files_only=True,
            revision=revision,
            token=token,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        cache_label = effective_cache_dir or "<default Hugging Face cache>"
        raise RuntimeError(
            f"Offline mode is enabled and {purpose} '{model_name_or_path}' is not installed under {cache_label}. "
            "Run the same command once with --install before retrying with --offline."
        ) from exc


def resolve_hf_file_path(
    repo_id: str,
    *,
    filename: str,
    purpose: str,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    token: Optional[str] = None,
    offline: Optional[bool] = None,
) -> str:
    effective_offline = is_offline_mode() if offline is None else offline
    effective_cache_dir = cache_dir or get_hf_cache_dir()
    local_files_only = effective_offline

    try:
        return _hf_hub_download(
            repo_id,
            filename=filename,
            cache_dir=effective_cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            token=token,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if not effective_offline:
            raise
        cache_label = effective_cache_dir or "<default Hugging Face cache>"
        raise RuntimeError(
            f"Offline mode is enabled and {purpose} '{repo_id}' is not installed under {cache_label}. "
            "Run the same command once with --install before retrying with --offline."
        ) from exc


def prefetch_hf_snapshot(
    repo_id: str,
    *,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    token: Optional[str] = None,
) -> str:
    return _snapshot_download(
        repo_id,
        cache_dir=cache_dir or get_hf_cache_dir(),
        local_files_only=False,
        revision=revision,
        token=token,
    )
