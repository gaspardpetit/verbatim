"""Helpers to ensure torchcodec can load FFmpeg DLLs for pyannote."""

# pylint: disable=import-outside-toplevel,broad-exception-caught

import logging
import os
from pathlib import Path
from typing import Optional

REQUIRED_PREFIXES = ("avcodec-", "avformat-", "avutil-")
LOG = logging.getLogger(__name__)


def _dir_has_ffmpeg_dlls(path: Path) -> bool:
    if not path.is_dir():
        return False

    names = {p.name.lower() for p in path.glob("av*.dll")}
    if not names:
        return False

    for prefix in REQUIRED_PREFIXES:
        if not any(name.startswith(prefix) for name in names):
            return False
    return True


def _add_dir(path: Path) -> str:
    """Register DLL dir on Windows and return the string path."""
    if os.name == "nt":
        os.add_dll_directory(str(path))
    return str(path)


def ensure_ffmpeg_for_torchcodec() -> Optional[str]:
    """
    Best-effort attempt to locate FFmpeg DLLs and make them loadable for torchcodec.

    Returns the directory used if found; raises RuntimeError if none found on Windows.
    """
    if os.name != "nt":
        return None

    tried = []

    env_dir = os.environ.get("FFMPEG_DLL_DIR")
    if env_dir:
        p = Path(env_dir)
        tried.append(("FFMPEG_DLL_DIR", str(p)))
        if _dir_has_ffmpeg_dlls(p):
            return _add_dir(p)

    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here / "ffmpeg",
        here / "bin",
        Path(os.getcwd()),
        Path(os.getcwd()) / "ffmpeg",
        Path(os.getcwd()) / "bin",
    ]
    for c in candidates:
        tried.append(("local", str(c)))
        if _dir_has_ffmpeg_dlls(c):
            return _add_dir(c)

    path_env = os.environ.get("PATH", "")
    for raw in path_env.split(os.pathsep):
        if not raw:
            continue
        p = Path(raw)
        tried.append(("PATH", str(p)))
        if _dir_has_ffmpeg_dlls(p):
            return _add_dir(p)

    details = "\n".join(f"  - {kind}: {p}" for kind, p in tried)
    raise RuntimeError(
        "Could not locate a usable FFmpeg shared build (DLLs) for TorchCodec.\n\n"
        "Expected to find at least avcodec-*.dll, avformat-*.dll, avutil-*.dll.\n\n"
        "Tried:\n"
        f"{details}\n\n"
        "To fix this, install FFmpeg 4–7 shared build and either:\n"
        "  - Set FFMPEG_DLL_DIR to the directory containing those DLLs, or\n"
        "  - Add that directory to PATH, or\n"
        "  - Copy the DLLs into an 'ffmpeg' folder next to this app.\n"
    )


def _register_torchcodec_audio_decoder() -> None:
    from torchcodec.decoders import AudioDecoder  # noqa: F401

    try:
        import pyannote.audio.core.io as pa_io

        setattr(pa_io, "AudioDecoder", AudioDecoder)  # pyright: ignore[reportPrivateImportUsage]
    except Exception as exc:  # pragma: no cover - best effort hook
        LOG.debug("Failed to register torchcodec AudioDecoder with pyannote: %s", exc)


def ensure_torchcodec_audio_decoder(context: str) -> None:
    """Ensure torchcodec is importable and wired into pyannote audio IO."""
    ensure_ffmpeg_for_torchcodec()

    try:
        _register_torchcodec_audio_decoder()
    except Exception:
        ensure_ffmpeg_for_torchcodec()
        try:
            _register_torchcodec_audio_decoder()
        except Exception as exc2:  # pragma: no cover - defensive import
            raise RuntimeError(
                f"{context} could not load torchcodec (FFmpeg dependency). "
                "Install FFmpeg shared libraries (4–7) and set FFMPEG_DLL_DIR or add them to PATH."
            ) from exc2
