import logging

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation
from pyannote.database.util import load_rttm


# Configure logger
LOG = logging.getLogger(__name__)


class Diarization:
    def __init__(self, device: str, huggingface_token: str):
        LOG.info("Initializing Diarization Pipeline.")
        self.huggingface_token = huggingface_token
        self.pipeline = Pipeline.from_pretrained(
            checkpoint_path="pyannote/speaker-diarization-3.1",
            use_auth_token=self.huggingface_token,
        )
        self.pipeline.instantiate({})

        self.pipeline.to(torch.device(device))

    def __enter__(self) -> "Diarization":
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        del self.pipeline
        return False

    @staticmethod
    def load_diarization(rttm_file: str):
        rttms = load_rttm(file_rttm=rttm_file)
        annotation: Annotation = next(iter(rttms.values()))
        return annotation

    # pylint: disable=unused-argument
    def compute_diarization(
        self, file_path: str, out_rttm_file: str = None, nb_speakers: int = None
    ) -> Annotation:
        if not out_rttm_file:
            out_rttm_file = "out.rttm"

        with ProgressHook() as hook:
            diarization = self.pipeline(file_path, hook=hook, num_speakers=nb_speakers)

        # dump the diarization output to disk using RTTM format
        with open(out_rttm_file, "w", encoding="utf-8") as rttm:
            diarization.write_rttm(rttm)

        return diarization
