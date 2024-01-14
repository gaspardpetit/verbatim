import logging
import numpy as np
from numpy import ndarray

from ..transcription import Transcription, Word, Utterance
from ..models.model_whisper import WhisperModel
from .transcribe_speech import TranscribeSpeech

LOG = logging.getLogger(__name__)

class TranscribeSpeechWhisper(TranscribeSpeech):
    """
    Implementation of TranscribeSpeech using the OpenAI Whisper ASR model.

    Attributes:
        None
    """

    def _get_utterance_from_segment(self, seg, language, speaker, speech_offset):
        """
        Create an Utterance object from a Whisper ASR model segment.

        Args:
            seg (dict): Whisper ASR model segment.
            language (str): Detected language.
            speaker (str): Speaker identifier.
            speech_offset (float): Offset in seconds from the beginning of the audio.

        Returns:
            Utterance: Utterance object created from the segment.
        """
        start = seg['start']
        end = seg['end']
        avg_logprob = seg['avg_logprob']
        no_speech_prob = seg['no_speech_prob']
        utterance = Utterance(
            start=speech_offset + start,
            end=speech_offset + end,
            language=language,
            confidence=np.exp(avg_logprob),
            speaker=speaker,
            words=[Word(text=word['word'], start=speech_offset + word['start'], end=speech_offset + word['end'],
                        confidence=(1 - no_speech_prob) * word['probability']) for word in seg['words']]
        )
        return utterance

    def execute_segment(self, speech_segment_float32_16khz: ndarray,
                        speaker: str = "speaker", speech_offset: float = 0,
                        language: str = None, prompt: str = "",
                        beams: int = 5, **kwargs: dict) -> Transcription:
        """
        Execute transcription on a speech segment using the OpenAI Whisper ASR model.

        Args:
            speaker (str): Speaker identifier.
            speech_offset (float): Offset in seconds from the beginning of the audio.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            language (str): Target language for transcription.
            prompt (str): Optional transcription prompt.
            beams (int): Number of beams for transcription.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription result for the given speech segment.
        """
        transcription = Transcription()

        WhisperModel.device = kwargs['device']
        verbose = (kwargs['log_level'] or logging.WARNING) >= logging.INFO
        model = WhisperModel().model
        whisper_transcription = model.transcribe(
            word_timestamps=True,
            audio=speech_segment_float32_16khz,
            initial_prompt=prompt,
            language=language,
            beam_size=beams,
            best_of=beams,
            verbose=verbose
        )
        language = whisper_transcription['language']
        for segment in whisper_transcription['segments']:
            utterance = self._get_utterance_from_segment(segment, language, speaker, speech_offset)
            transcription.append(utterance)
            LOG.info(f"silent={segment['no_speech_prob']:.2%}|{utterance.get_colour_text()}")
        return transcription
