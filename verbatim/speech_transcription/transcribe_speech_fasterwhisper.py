import logging
import numpy as np
from numpy import ndarray

from ..transcription import Transcription, Word, Utterance
from ..models.model_fasterwhisper import FasterWhisperModel
from .transcribe_speech import TranscribeSpeech


LOG = logging.getLogger(__name__)

class TranscribeSpeechFasterWhisper(TranscribeSpeech):
    """
    Implementation of TranscribeSpeech using FasterWhisper model.

    Attributes:
        None
    """

    def _get_utterance_from_segment(self, segment, info, speaker, speech_offset):
        """
        Extracts Utterance information from a segment.

        Args:
            segment: Segment information from FasterWhisper model.
            info: Additional information about the segment.
            speaker (str): The speaker identifier.
            speech_offset (float): The offset in seconds from the beginning of the audio.

        Returns:
            Utterance: Utterance object containing information about the transcribed segment.
        """

        start = speech_offset + segment.start
        end = speech_offset + segment.end
        silence_prob = segment.no_speech_prob
        utterance = Utterance(
            start=start,
            end=end,
            language=info.language,
            confidence=np.exp(segment.avg_logprob),
            speaker=speaker,
            words=[Word(text=w.word, start=speech_offset + w.start, end=speech_offset + w.end,
                        confidence=(1 - silence_prob) * w.probability) for w in segment.words]
        )
        return utterance

    def execute_segment(self, speaker: str, speech_offset: float, speech_segment_float32_16khz: ndarray, language=None,
                        prompt: str = "", beams: int = 5, **kwargs: dict) -> Transcription:
        """
        Executes transcription on a speech segment using FasterWhisper model.

        Args:
            speaker (str): The speaker identifier.
            speech_offset (float): The offset in seconds from the beginning of the audio.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            language (str): Target language for transcription.
            prompt (str): Optional transcription prompt.
            beams (int): Number of beams for decoding.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription object containing the transcribed information.
        """
        transcription = Transcription()
        FasterWhisperModel.device = kwargs['device']
        model = FasterWhisperModel().model

        segments, info = model.transcribe(
            word_timestamps=True,
            audio=speech_segment_float32_16khz,
            initial_prompt=prompt,
            language=language,
            beam_size=beams,
            best_of=beams,
            patience=1,
            vad_filter=True
        )
        for segment in segments:
            utterance = self._get_utterance_from_segment(segment, info, speaker, speech_offset)
            transcription.append(utterance)
            LOG.info(f"[silent={segment.no_speech_prob:.2%};" +
                     f"temp={segment.temperature:.2f};" + 
                     f"comp={segment.compression_ratio:.2%}]" + 
                     f"{utterance.get_colour_text()}")
        return transcription

    def detect_language(self, speaker: str, speech_offset: float, speech_segment_float32_16khz: ndarray,
                        languages=None, **kwargs: dict) -> Transcription:
        """
        Detects language in a speech segment using FasterWhisper model.

        Args:
            speaker (str): The speaker identifier.
            speech_offset (float): The offset in seconds from the beginning of the audio.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            language_file (str): File path for saving language information.
            languages (list): List of target languages.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription object containing the detected language information.
        """
        transcription = Transcription()

        FasterWhisperModel.device = kwargs['device']
        model = FasterWhisperModel().model

        features = model.feature_extractor(speech_segment_float32_16khz)

        segment = features[:, : model.feature_extractor.nb_max_frames]
        encoder_output = model.encode(segment)
        # results is a list of tuple[str, float] with language names and
        # probabilities.
        results = model.model.detect_language(encoder_output)[0]
        # Parse language names to strip out markers
        all_language_probs = [(token[2:-2], prob) for (token, prob) in results]
        detected_languages = {}
        for l, p in all_language_probs:
            detected_languages[l] = p

        best_language = None
        best_language_prob = 0
        if languages is not None and len(languages) > 0:
            for lang in languages:
                if lang in detected_languages:
                    if detected_languages[lang] > best_language_prob:
                        best_language_prob = detected_languages[lang]
                        best_language = lang

        if best_language is None:
            best_language = max(detected_languages, key=detected_languages.get)
            best_language_prob = detected_languages[best_language]

        utterance = Utterance(
            start=speech_offset,
            end=speech_offset + len(speech_segment_float32_16khz) / 16000,
            language=best_language,
            confidence=best_language_prob,
            speaker=speaker,
            words=[])
        transcription.append(utterance)
        LOG.info(utterance.get_colour_text())
        return transcription
    