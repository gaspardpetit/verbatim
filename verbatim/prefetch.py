import logging
import os
import platform
import re
import sys
from getpass import getpass
from typing import Optional

# pylint: disable=broad-exception-caught

# Note on import order:
# We import huggingface_hub inside prefetch() after applying cache/offline env so that
# the library picks up VERBATIM/HF_* cache dirs. Importing at module level would make
# it stick to the user's default ~/.cache path.

try:
    import whisper as openai_whisper  # type: ignore
except ImportError:  # pragma: no cover
    openai_whisper = None  # type: ignore

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
        # Prefetch requires network; make sure offline flags do not linger
        os.environ["VERBATIM_OFFLINE"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

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
    LOG.debug(
        "Cache env: VERBATIM_MODEL_CACHE=%s HF_HOME=%s HUGGINGFACE_HUB_CACHE=%s",
        os.getenv("VERBATIM_MODEL_CACHE"),
        os.getenv("HF_HOME"),
        os.getenv("HUGGINGFACE_HUB_CACHE"),
    )


def _hf_models_dir(repo_id: str) -> str:
    """Return the Hugging Face hub models directory name for a repo id.

    Example: "org/name" -> "models--org--name"
    """
    parts = repo_id.split("/")
    if len(parts) != 2:
        # Best-effort; fallback to replacing separators
        safe = repo_id.replace("/", "--")
        return f"models--{safe}"
    return f"models--{parts[0]}--{parts[1]}"


def _resolve_hf_revision(repo_id: str, *, local_dir: Optional[str] = None) -> str:
    """Best-effort resolution of a pinned revision for a HF repo.

    Tries to read a cached commit hash from refs/main if available in either:
      - the provided local_dir (when using snapshot_download local_dir), or
      - the global HF cache pointed by HUGGINGFACE_HUB_CACHE/HF_HOME.

    Falls back to 'main' when no hash is found. Bandit B615 accepts the presence
    of the 'revision' argument; when available we prefer an actual SHA.
    """
    # Regex for 40-hex commit SHA
    sha_re = re.compile(r"^[0-9a-f]{40}$")

    candidates = []
    models_dirname = _hf_models_dir(repo_id)

    if local_dir:
        candidates.append(os.path.join(local_dir, models_dirname, "refs", "main"))

    # Global hub cache (preferred path when not using local_dir)
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if not hub_cache:
        hf_home = os.getenv("HF_HOME")
        if hf_home:
            hub_cache = os.path.join(hf_home, "hub")
    if hub_cache:
        candidates.append(os.path.join(hub_cache, models_dirname, "refs", "main"))

    for ref_path in candidates:
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                ref = f.read().strip()
            if sha_re.match(ref):
                return ref
        except OSError:
            continue

    return "main"


def prefetch(
    *,
    model_cache_dir: Optional[str],
    whisper_size: str = "large-v3",
    include_pyannote: bool = True,
    include_whisper_openai: bool = True,
    include_faster_whisper: bool = True,
    # Defer platform checks to runtime to keep import safe on Windows
    include_mlx_whisper: Optional[bool] = None,
) -> None:
    """Prefetch commonly used models into the cache.

    This function attempts to only download/copy into the provided cache directory,
    so later runs can use --offline.
    """
    # Compute default for MLX Whisper lazily to avoid os.uname() at import time
    if include_mlx_whisper is None:
        include_mlx_whisper = sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")

    apply_cache_env(model_cache_dir, offline=False)

    # Lazy-import huggingface_hub after env is configured so it uses our cache dirs
    # pylint: disable=import-outside-toplevel
    try:  # type: ignore
        from huggingface_hub import snapshot_download  # type: ignore
        from huggingface_hub.errors import HfHubHTTPError  # type: ignore
        from huggingface_hub.utils import LocalEntryNotFoundError  # type: ignore
    except ImportError:  # pragma: no cover
        snapshot_download = None  # type: ignore
        HfHubHTTPError = Exception  # type: ignore
        LocalEntryNotFoundError = Exception  # type: ignore

    # 1) Hugging Face models via huggingface_hub
    if snapshot_download is None:  # pragma: no cover - optional path
        LOG.warning("huggingface_hub not available: cannot prefetch HF-hosted models")

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")

    # Pyannote diarization/separation
    if include_pyannote and snapshot_download is not None:
        for repo in ("pyannote/speaker-diarization-3.1", "pyannote/speech-separation-ami-1.0"):
            rev = _resolve_hf_revision(repo)
            # Fast path: try local only to avoid network churn if already cached
            try:
                local_path = snapshot_download(
                    repo_id=repo,
                    token=hf_token,
                    local_files_only=True,
                    revision=rev,
                    cache_dir=hub_cache,
                )
                LOG.info("Already cached: %s (rev=%s) at %s", repo, rev, local_path)
                continue
            except LocalEntryNotFoundError:
                pass

            LOG.info(f"Prefetching HF repo: {repo}")
            try:
                local_path = snapshot_download(
                    repo_id=repo,
                    token=hf_token,
                    local_files_only=False,
                    revision=rev,
                    cache_dir=hub_cache,
                )
                LOG.info("Downloaded: %s (rev=%s) to %s", repo, rev, local_path)
                continue
            except HfHubHTTPError as e_first:  # pragma: no cover
                # If unauthorized and interactive, prompt for token and retry once
                err_txt = str(e_first)
                status = getattr(getattr(e_first, "response", None), "status_code", None)
                lc = err_txt.lower()
                unauthorized = (
                    status in (401, 403)
                    or "401" in lc
                    or "403" in lc
                    or "forbidden" in lc
                    or "unauthorized" in lc
                    or "unauth" in lc
                    or "restricted" in lc
                    or "gated" in lc
                )

                if unauthorized and sys.stdin.isatty() and sys.stdout.isatty():
                    print("Pyannote models are gated and require a Hugging Face token.")
                    print("Create one at https://huggingface.co/settings/tokens and ensure gated model access.")
                    try:
                        entered = getpass("Enter HUGGINGFACE_TOKEN (starts with hf_): ")
                    except (EOFError, KeyboardInterrupt):  # pragma: no cover
                        entered = ""
                    if entered:
                        hf_token = entered.strip()
                        os.environ["HUGGINGFACE_TOKEN"] = hf_token
                        print("Token received. Tip: export HUGGINGFACE_TOKEN to avoid prompts next time.")
                        try:
                            local_path = snapshot_download(
                                repo_id=repo,
                                token=hf_token,
                                local_files_only=False,
                                revision=rev,
                                cache_dir=hub_cache,
                            )
                            LOG.info("Downloaded after auth: %s (rev=%s) to %s", repo, rev, local_path)
                            continue
                        except HfHubHTTPError as e_retry:  # pragma: no cover
                            LOG.warning(f"Failed to prefetch {repo} after prompt: {e_retry}")
                            continue
                else:
                    LOG.info("Non-interactive terminal; skipping prompt for HF token")

                LOG.warning(f"Failed to prefetch {repo}: {e_first}")

    # MLX Whisper models (macOS/Apple Silicon) hosted on HF
    if include_mlx_whisper and snapshot_download is not None:
        mlx_repo = f"mlx-community/whisper-{whisper_size}-mlx"
        mlx_rev = _resolve_hf_revision(mlx_repo)
        try:
            LOG.info(f"Prefetching HF repo: {mlx_repo} (rev={mlx_rev})")
            local_path = snapshot_download(
                repo_id=mlx_repo,
                token=hf_token,
                local_files_only=False,
                revision=mlx_rev,
                cache_dir=hub_cache,
            )
            LOG.info("Downloaded: %s (rev=%s) to %s", mlx_repo, mlx_rev, local_path)
        except HfHubHTTPError as e:  # pragma: no cover
            LOG.warning(f"Failed to prefetch {mlx_repo}: {e}")

    # 2) Faster-Whisper: populate cache without heavy initialization
    if include_faster_whisper:
        try:
            cache_root = os.getenv("VERBATIM_MODEL_CACHE")
            download_root = os.path.join(cache_root, "faster-whisper") if cache_root else None
            fw_repo = f"Systran/faster-whisper-{whisper_size}"
            fw_rev = _resolve_hf_revision(fw_repo, local_dir=download_root)

            if download_root:
                os.makedirs(download_root, exist_ok=True)
                if snapshot_download is not None:
                    try:
                        # Check if already present in the same cache layout runtime will use
                        snapshot_download(
                            repo_id=fw_repo,
                            local_files_only=True,
                            revision=fw_rev,
                            cache_dir=download_root,
                        )
                        LOG.info("Already cached: faster-whisper (%s)", whisper_size)
                    except LocalEntryNotFoundError:
                        LOG.info(f"Prefetching faster-whisper model: {whisper_size}")
                        # Populate download_root as a full HF cache so runtime offline lookup succeeds
                        local_path = snapshot_download(
                            repo_id=fw_repo,
                            local_files_only=False,
                            revision=fw_rev,
                            cache_dir=download_root,
                        )
                        LOG.info("Downloaded: %s (rev=%s) to %s", fw_repo, fw_rev, local_path)
            else:
                if snapshot_download is not None:
                    LOG.info(f"Prefetching faster-whisper model: {whisper_size}")
                    # No specific download_root: populate global project HF cache
                    local_path = snapshot_download(
                        repo_id=fw_repo,
                        local_files_only=False,
                        revision=fw_rev,
                        cache_dir=hub_cache,
                    )
                    LOG.info("Downloaded (HF cache): %s (rev=%s) to %s", fw_repo, fw_rev, local_path)
        except (OSError, HfHubHTTPError) as e:  # pragma: no cover
            LOG.warning(f"Failed to prefetch faster-whisper {whisper_size}: {e}")

    # 3) OpenAI Whisper (downloads .pt to WHISPER_CACHE_DIR)
    if include_whisper_openai:
        whisper_cache = os.getenv("WHISPER_CACHE_DIR", os.path.join(os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "whisper"))
        try:
            os.makedirs(whisper_cache, exist_ok=True)
        except OSError:  # pragma: no cover
            pass
        target = os.path.join(whisper_cache, f"{whisper_size}.pt")
        if os.path.exists(target):
            LOG.info("Already cached: openai/whisper (%s)", whisper_size)
        else:
            if openai_whisper is None:  # pragma: no cover
                LOG.warning("openai-whisper not available: cannot prefetch %s", whisper_size)
            else:
                LOG.info(f"Prefetching OpenAI whisper model: {whisper_size}")
                _ = openai_whisper.load_model(whisper_size, device="cpu")

    # 4) Voice isolation: provide guidance; download may be manual depending on backend
    cache_root = os.getenv("VERBATIM_MODEL_CACHE")
    if cache_root:
        iso_dir = os.path.join(cache_root, "audio-separator")
        os.makedirs(iso_dir, exist_ok=True)
        LOG.info(f"If needed, place MDX checkpoint (e.g., MDX23C-8KFFT-InstVoc_HQ_2.ckpt) under: {iso_dir} to enable offline isolation")
