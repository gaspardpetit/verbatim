"""Model prefetching for diarization/separation artifacts."""

import logging
from typing import Optional

LOG = logging.getLogger(__name__)


def prefetch_diarization_models(hf_token: Optional[str] = None, cache_dir: Optional[str] = None, offline: bool = False) -> None:
    """Prefetch diarization/separation models (pyannote).

    Expects cache/offline environment variables to be set by the caller.
    """
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        from huggingface_hub.errors import HfHubHTTPError  # type: ignore
        from huggingface_hub.utils import LocalEntryNotFoundError  # type: ignore
    except ImportError:  # pragma: no cover
        LOG.warning("huggingface_hub not available: cannot prefetch diarization models")
        return

    repos = ("pyannote/speaker-diarization-3.1", "pyannote/speech-separation-ami-1.0")
    for repo in repos:
        try:
            # Local-only try first
            local_path = snapshot_download(repo_id=repo, token=hf_token, local_files_only=True, cache_dir=cache_dir)
            LOG.info("Already cached diarization repo: %s at %s", repo, local_path)
            continue
        except LocalEntryNotFoundError:
            if offline:
                LOG.warning("Offline mode set and %s not cached; skipping download.", repo)
                continue
        except Exception as exc:  # pragma: no cover - defensive
            LOG.debug("Local cache check failed for %s: %s", repo, exc)

        if offline:
            LOG.info("Offline mode; skipping download for %s", repo)
            continue

        LOG.info("Prefetching diarization model: %s", repo)
        try:
            local_path = snapshot_download(repo_id=repo, token=hf_token, local_files_only=False, cache_dir=cache_dir)
            LOG.info("Downloaded %s to %s", repo, local_path)
        except HfHubHTTPError as exc:
            LOG.warning("Failed to prefetch %s: %s", repo, exc)
