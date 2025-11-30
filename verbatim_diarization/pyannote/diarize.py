import os
from typing import Optional

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation

from verbatim_diarization.diarize.base import DiarizationStrategy
from verbatim_rttm import Annotation as RTTMAnnotation
from verbatim_rttm import AudioRef, Segment, write_vttm


class PyAnnoteDiarization(DiarizationStrategy):
    def __init__(self, device: str, huggingface_token: str):
        self.device = device
        self.huggingface_token = huggingface_token
        self.pipeline = None

    def initialize_pipeline(self):
        """Lazy initialization of PyAnnote pipeline"""
        if self.pipeline is None:
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=self.huggingface_token)
        if self.pipeline is None:
            raise RuntimeError("PyAnnote pipeline failed to initialize")
        self.pipeline.instantiate({})
        self.pipeline.to(torch.device(self.device))

    def compute_diarization(
        self, file_path: str, out_rttm_file: Optional[str] = None, out_vttm_file: Optional[str] = None, nb_speakers: Optional[int] = None, **kwargs
    ) -> Annotation:
        """
        Compute diarization using PyAnnote.

        Additional kwargs:
            nb_speakers: Optional number of speakers
        """
        try:
            # pyannote.audio 4.x requires torchcodec for audio decoding
            pass  # type: ignore[unused-import]
        except Exception as exc:  # pragma: no cover - defensive import
            raise RuntimeError("""
                pyannote diarization requires torchcodec for audio decoding;
                install torchcodec (and compatible torch) or switch diarization_strategy.
                """) from exc
        self.initialize_pipeline()
        pipeline = self.pipeline
        if pipeline is None:
            raise RuntimeError("PyAnnote pipeline failed to initialize")
        with ProgressHook() as hook:
            diarization = pipeline(file_path, hook=hook, num_speakers=nb_speakers)

        # pyannote.audio 4.x returns a DiarizeOutput with a speaker_diarization field
        if hasattr(diarization, "speaker_diarization"):
            diarization_annotation = diarization.speaker_diarization  # type: ignore[attr-defined]
        else:
            diarization_annotation = diarization

        if out_rttm_file:
            self.save_rttm(diarization_annotation, out_rttm_file)

        if out_vttm_file:
            os.makedirs(os.path.dirname(out_vttm_file) or ".", exist_ok=True)
            uri = os.path.splitext(os.path.basename(file_path))[0]
            segments = [
                Segment(start=segment.start, end=segment.end, speaker=str(label), file_id=uri)
                for segment, _track, label in diarization_annotation.itertracks(yield_label=True)
            ]
            rttm_ann = RTTMAnnotation(segments=segments, file_id=uri)
            write_vttm(out_vttm_file, audio=[AudioRef(id=uri, path=file_path)], annotation=rttm_ann)

        return diarization_annotation
