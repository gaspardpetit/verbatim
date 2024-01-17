﻿"""
__init__.py
"""
import warnings

# see https://github.com/pyannote/pyannote-audio/issues/1576
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")

# see https://github.com/asteroid-team/torch-audiomentations/issues/172
warnings.filterwarnings("ignore", category=UserWarning, module=r"torch_audiomentations\.utils\.io")

# see https://github.com/asteroid-team/torch-audiomentations/issues/172
warnings.filterwarnings("ignore", category=UserWarning, module=r".*")

# pylint: disable=wrong-import-position
from .context import Context
from .pipeline import Pipeline

__version__ = "0.1.5"
