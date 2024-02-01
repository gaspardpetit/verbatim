import os
import logging
import torchaudio
import numpy as np

from speechbrain.pretrained import VAD
from pyannote.core import Annotation, Segment

from ..wav_conversion import ConvertToWav
from .diarize_speakers import DiarizeSpeakers

LOG = logging.getLogger(__name__)


class DiarizeSpeakersSpeechBrain(DiarizeSpeakers):
    """
    Diarization implementation using SpeechBrain for speaker segmentation.

    This class inherits from DiarizeSpeakers and implements diarization using SpeechBrain for speaker segmentation.

    Attributes:
        None
    """

    def diarize_on_silences(self, audio_path: str, model_speechbrain_vad:str, **kwargs:dict) -> Annotation:
        """
        Diarize speakers based on silences using SpeechBrain.

        Args:
            audio_file (str): Path to the input audio file.

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # Set up temporary directory for VAD model
        tmpdir = os.path.join(kwargs['work_directory_path'], "tmpdir")

        # Load VAD model from SpeechBrain
        vad_model = VAD.from_hparams(
            source=model_speechbrain_vad,
            savedir=tmpdir,
            run_opts={"device": kwargs['device']}
        )

        # Perform VAD
        boundaries = vad_model.get_speech_segments(audio_path)

        # Create a Pyannote Annotation object for diarization
        diarization = Annotation()

        # Add speaker segments to the diarization annotation
        for i in range(0, len(boundaries), 2):
            if i+1 < len(boundaries):
                diarization[Segment(float(boundaries[i][0]), float(boundaries[i + 1][1]))] = "speaker"
            else:
                diarization[Segment(float(boundaries[i][0]), float(boundaries.data[0][1]))] = "speaker"

        # Upsample boundaries and save the VAD result as a new audio file
        upsampled_boundaries = vad_model.upsample_boundaries(boundaries=boundaries, audio_file=audio_path)
        torchaudio.save(f"{audio_path}_vad.wav", upsampled_boundaries.cpu(), 16000)

        return diarization

    @staticmethod
    def pad_audio_to_duration(audio_samples, target_duration_seconds, sampling_rate=16000):
        # Calculate the target length in samples
        target_length_samples = int(target_duration_seconds * sampling_rate)

        # Calculate the current length of the audio
        current_length_samples = len(audio_samples)

        # Calculate the amount of padding needed
        padding_needed = target_length_samples - current_length_samples
        if padding_needed <= 0:
            return audio_samples

        # Pad the audio samples with zeros (silence)
        padded_audio = np.pad(audio_samples, (0, padding_needed), mode='constant', constant_values=0)

        return padded_audio

    def execute(self, voice_file_path: str, diarization_file: str,
                min_speakers: int = 1, max_speakers: int = None,
                model_speechbrain_vad:str  = None, **kwargs: dict) -> Annotation:
        """
        Execute the diarization process using SpeechBrain.

        Args:
            voice_file_path (str): Path to the input audio file.
            diarization_file (str): Path to the output RTTM (Rich Transcription Time Marked) file.
            min_speakers (int, optional): Minimum number of expected speakers. Default is 1.
            max_speakers (int, optional): Maximum number of expected speakers. Default is None (unbounded).
            **kwargs (dict): Additional parameters (not used in this method).

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # Perform diarization based on silences using SpeechBrain
        diarization: Annotation = self.diarize_on_silences(audio_path=voice_file_path,
                                                           model_speechbrain_vad=model_speechbrain_vad)

        # Write the diarization result to the output RTTM file
        with open(diarization_file, "w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        # Log the diarization result for information
        LOG.info(diarization)

        return diarization
