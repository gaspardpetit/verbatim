"""
__init__.py
"""
import warnings
import logging

# see https://github.com/asteroid-team/torch-audiomentations/issues/172
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch_audiomentations\.utils\.io")

# see https://github.com/asteroid-team/torch-audiomentations/issues/172
warnings.filterwarnings("ignore", category=UserWarning, module=r".*")

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"silero_vad\.model")

# disable INFO:speechbrain.utils.quirks:Applied quirks (see `speechbrain.utils.quirks`): [disable_jit_profiling, allow_tf32]
#         INFO:speechbrain.utils.quirks:Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): []
logging.getLogger("speechbrain.utils.quirks").setLevel(logging.WARNING)

__version__ = "1.0.1"
