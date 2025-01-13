import logging
from typing import Tuple, Dict, Optional

import numpy as np
import scipy.io.wavfile
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation

from ..audio.audio import wav_to_int16

# Configure logger
LOG = logging.getLogger(__name__)


class SpeakerSeparation:
    def __init__(self, device: str, huggingface_token: str):
        LOG.info("Initializing Separation Pipeline.")
        self.huggingface_token = huggingface_token
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speech-separation-ami-1.0",
            use_auth_token=self.huggingface_token,
        )
        hyper_parameters = {
            "segmentation": {"min_duration_off": 0.0, "threshold": 0.82},
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
                "threshold": 0.68,
            },
            "separation": {
                "leakage_removal": True,
                "asr_collar": 0.32,
            },
        }

        self.pipeline.instantiate(hyper_parameters)

        self.pipeline.to(torch.device(device))

    def __enter__(self) -> "SpeakerSeparation":
        return self

    def __exit__(self, ex_type, ex_value, ex_traceback):
        del self.pipeline
        return False

    # pylint: disable=unused-argument
    def separate_speakers(
        self,
        *,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_speaker_wav_prefix="",
        nb_speakers: Optional[int] = None,
    ) -> Tuple[Annotation, Dict[str, str]]:
        if not out_rttm_file:
            out_rttm_file = "out.rttm"

        with ProgressHook() as hook:
            diarization, sources = self.pipeline(file_path, hook=hook)

        # dump the diarization output to disk using RTTM format
        with open(out_rttm_file, "w", encoding="utf-8") as rttm:
            diarization.write_rttm(rttm)

        # dump sources to disk as SPEAKER_XX.wav files
        speaker_wav_files = {}
        for s, speaker in enumerate(diarization.labels()):
            if s < sources.data.shape[1]:
                speaker_data = sources.data[:, s]
                if speaker_data.dtype != np.int16:
                    speaker_data = wav_to_int16(speaker_data)
                file_name = f"{out_speaker_wav_prefix}-{speaker}.wav" if out_speaker_wav_prefix else f"{speaker}.wav"
                speaker_wav_files[speaker] = file_name
                scipy.io.wavfile.write(file_name, 16000, speaker_data)
            else:
                LOG.debug(f"Skipping speaker {s} as it is out of bounds.")
        return diarization, speaker_wav_files
