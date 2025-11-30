from typing import Optional

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation

from .base import DiarizationStrategy


class PyAnnoteDiarization(DiarizationStrategy):
    def __init__(self, device: str, huggingface_token: str):
        self.device = device
        self.huggingface_token = huggingface_token
        self.pipeline = None

    def initialize_pipeline(self):
        """Lazy initialization of PyAnnote pipeline"""
        if self.pipeline is None:
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=self.huggingface_token)
            self.pipeline.instantiate({})
            self.pipeline.to(torch.device(self.device))

    def compute_diarization(self, file_path: str, out_rttm_file: Optional[str] = None, nb_speakers: Optional[int] = None, **kwargs) -> Annotation:
        """
        Compute diarization using PyAnnote.

        Additional kwargs:
            nb_speakers: Optional number of speakers
        """
        self.initialize_pipeline()
        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("PyAnnote pipeline failed to initialize")
        with ProgressHook() as hook:
            diarization = pipeline(file_path, hook=hook, num_speakers=nb_speakers)

        if out_rttm_file:
            self.save_rttm(diarization, out_rttm_file)

        return diarization
