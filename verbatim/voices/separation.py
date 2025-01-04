import logging
from typing import Union

import numpy as np
import scipy.io.wavfile
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation
from pyannote.database.util import load_rttm

from ..audio.audio import wav_to_int16

# Configure logger
LOG = logging.getLogger(__name__)

class Diarization:
    def __init__(self, device:str, huggingface_token:str):
        LOG.info("Initializing Diarization Pipeline.")
        self.huggingface_token = huggingface_token
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speech-separation-ami-1.0",
            use_auth_token=self.huggingface_token
        )
        hyper_parameters =         {
                "segmentation": {
                "min_duration_off": 0.0,
                "threshold": 0.82
                },
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": 15,
                    "threshold": 0.68,
                },
                "separation": {
                    "leakage_removal": True,
                    "asr_collar": 0.32,
                }
            }

        self.pipeline.instantiate(hyper_parameters)
        self.pipeline.to(torch.device(device))

    @staticmethod
    def load_diarization(rttm_file:str):
        rttms = load_rttm(file_rttm=rttm_file)
        annotation:Annotation = next(iter(rttms.values()))
        return annotation

    # pylint: disable=unused-argument
    def compute_diarization(self, file_path:str, out_rttm_file:Union[None,str]=None, nb_speakers:Union[None,int]=None) -> Annotation:
        if not out_rttm_file:
            out_rttm_file = "out.rttm"

        sources = None
        with ProgressHook() as hook:
            diarization, sources = self.pipeline(file_path, hook=hook)

        # dump the diarization output to disk using RTTM format
        with open(out_rttm_file, "w", encoding="utf-8") as rttm:
            diarization.write_rttm(rttm)

        # dump sources to disk as SPEAKER_XX.wav files
        for s, speaker in enumerate(diarization.labels()):
            if s < sources.data.shape[1]:
                speaker_data = sources.data[:, s]
                if speaker_data.dtype != np.int16:
                    speaker_data = wav_to_int16(speaker_data)
                scipy.io.wavfile.write(f'{speaker}.wav', 16000, speaker_data)
            else:
                LOG.debug(f"Skipping speaker {s} as it is out of bounds.")
        return diarization
