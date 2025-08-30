"""
__init__.py
"""

import warnings
import logging
from matplotlib._api.deprecation import MatplotlibDeprecationWarning

# see https://github.com/asteroid-team/torch-audiomentations/issues/172
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch_audiomentations\.utils\.io")

# see https://github.com/asteroid-team/torch-audiomentations/issues/172
warnings.filterwarnings("ignore", category=UserWarning, module=r".*")

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"silero_vad\.model")

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# disable INFO:speechbrain.utils.quirks:Applied quirks (see `speechbrain.utils.quirks`): [disable_jit_profiling, allow_tf32]
#         INFO:speechbrain.utils.quirks:Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): []
logging.getLogger("speechbrain.utils.quirks").setLevel(logging.WARNING)

try:
    # Prefer version file generated at build time by hatch-vcs
    from ._version import __version__  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from importlib.metadata import (
            version as _pkg_version,
            PackageNotFoundError,
        )  # pyright: ignore[reportMissingImports]
        __version__ = _pkg_version("verbatim")
    except PackageNotFoundError:
        __version__ = "0.0.0-dev"
