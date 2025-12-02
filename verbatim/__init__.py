"""
__init__.py
"""

import warnings

# Avoid importing matplotlib on startup; fall back to DeprecationWarning for filtering.
MatplotlibDeprecationWarning = DeprecationWarning

# see https://github.com/asteroid-team/torch-audiomentations/issues/172
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch_audiomentations\.utils\.io")

# see https://github.com/asteroid-team/torch-audiomentations/issues/172
warnings.filterwarnings("ignore", category=UserWarning, module=r".*")

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"silero_vad\.model")

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

try:
    # Prefer version file generated at build time by hatch-vcs
    from ._version import __version__  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from importlib.metadata import (
            PackageNotFoundError,
        )
        from importlib.metadata import (
            version as _pkg_version,
        )  # pyright: ignore[reportMissingImports]

        __version__ = _pkg_version("verbatim")
    except PackageNotFoundError:
        __version__ = "0.0.0-dev"
