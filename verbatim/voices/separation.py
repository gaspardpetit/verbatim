import logging
from typing import Tuple, Dict, Optional

import numpy as np
import scipy.io.wavfile
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core.annotation import Annotation

from ..audio.audio import wav_to_int16
from .diarize.factory import create_diarizer

# Configure logger
LOG = logging.getLogger(__name__)


class SpeakerSeparation:
    def __init__(self, device: str, huggingface_token: str):
        LOG.info("Initializing Separation Pipeline.")
        self.device = device
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
        diarization_strategy: str = "pyannote",
    ) -> Tuple[Annotation, Dict[str, str]]:
        """
        Separate speakers in an audio file.

        Args:
            file_path: Path to input audio file
            out_rttm_file: Path to output RTTM file
            out_speaker_wav_prefix: Prefix for output WAV files
            nb_speakers: Optional number of speakers
            diarization_strategy: Diarization strategy to use ('pyannote' or 'stereo')

        Returns:
            Tuple of (diarization annotation, dictionary mapping speaker IDs to WAV files)
        """
        if not out_rttm_file:
            out_rttm_file = "out.rttm"

        # For stereo strategy, we might want to handle separation differently
        if diarization_strategy == "stereo":
            # For stereo files, we can simply split the channels
            sample_rate, audio_data = scipy.io.wavfile.read(file_path)
            if audio_data.ndim != 2 or audio_data.shape[1] != 2:
                raise ValueError("Stereo separation requires stereo audio input")

            # Create diarization annotation
            diarizer = create_diarizer(strategy="stereo", device=self.device, huggingface_token=self.huggingface_token)
            diarization = diarizer.compute_diarization(file_path=file_path, out_rttm_file=out_rttm_file, nb_speakers=nb_speakers)

            # Split channels into separate files
            speaker_wav_files = {}
            for channel, speaker in enumerate(["SPEAKER_0", "SPEAKER_1"]):
                channel_data = audio_data[:, channel]
                if channel_data.dtype != np.int16:
                    channel_data = wav_to_int16(channel_data)
                file_name = f"{out_speaker_wav_prefix}-{speaker}.wav" if out_speaker_wav_prefix else f"{speaker}.wav"
                speaker_wav_files[speaker] = file_name
                scipy.io.wavfile.write(file_name, sample_rate, channel_data)

            return diarization, speaker_wav_files

        else:
            # Use PyAnnote's neural separation for mono files
            with ProgressHook() as hook:
                diarization, sources = self.pipeline(file_path, hook=hook)

            # Save diarization to RTTM file
            with open(out_rttm_file, "w", encoding="utf-8") as rttm:
                diarization.write_rttm(rttm)

            # Save separated sources to WAV files
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
