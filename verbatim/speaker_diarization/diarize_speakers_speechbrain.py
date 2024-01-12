import logging
import torchaudio

from speechbrain.pretrained import VAD
from pyannote.core import Annotation, Segment

from .diarize_speakers import DiarizeSpeakers

LOG = logging.getLogger(__name__)


class DiarizeSpeakersSpeechBrain(DiarizeSpeakers):
    """
    Diarization implementation using SpeechBrain for speaker segmentation.

    This class inherits from DiarizeSpeakers and implements diarization using SpeechBrain for speaker segmentation.

    Attributes:
        None
    """

    def diarize_on_silences(self, audio_file: str) -> Annotation:
        """
        Diarize speakers based on silences using SpeechBrain.

        Args:
            audio_file (str): Path to the input audio file.

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # Set up temporary directory for VAD model
        tmpdir = "tmpdir"

        # Load VAD model from SpeechBrain
        vad_model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir=tmpdir,
            run_opts={"device": "cuda"}
        )

        # Perform VAD
        boundaries = vad_model.get_speech_segments(audio_file)

        # Create a Pyannote Annotation object for diarization
        diarization = Annotation()

        # Add speaker segments to the diarization annotation
        for i in range(0, len(boundaries), 2):
            if i+1 < len(boundaries):
                diarization[Segment(float(boundaries[i][0]), float(boundaries[i + 1][1]))] = "speaker"
            else:
                diarization[Segment(float(boundaries[i][0]), float(len(audio_file)/16000))] = "speaker"

        # Upsample boundaries and save the VAD result as a new audio file
        upsampled_boundaries = vad_model.upsample_boundaries(boundaries=boundaries, audio_file=audio_file)
        torchaudio.save(f"{audio_file}_vad.wav", upsampled_boundaries.cpu(), 16000)

        return diarization

    def execute(self, audio_file: str, rttm_file: str, min_speakers: int = 1, max_speakers: int = None,
                **kwargs: dict) -> Annotation:
        """
        Execute the diarization process using SpeechBrain.

        Args:
            audio_file (str): Path to the input audio file.
            rttm_file (str): Path to the output RTTM (Rich Transcription Time Marked) file.
            min_speakers (int, optional): Minimum number of expected speakers. Default is 1.
            max_speakers (int, optional): Maximum number of expected speakers. Default is None (unbounded).
            **kwargs (dict): Additional parameters (not used in this method).

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        # Perform diarization based on silences using SpeechBrain
        diarization: Annotation = self.diarize_on_silences(audio_file)

        # Write the diarization result to the output RTTM file
        with open(rttm_file, "w", encoding="utf-8") as f:
            diarization.write_rttm(f)

        # Log the diarization result for information
        LOG.info(diarization)

        return diarization
