from ..transcription import Transcription, Utterance
from .detect_language import DetectLanguage
from ..models.model_fasterwhisper import FasterWhisperModel

from numpy import ndarray
import logging

LOG = logging.getLogger(__name__)


class DetectLanguageFasterWhisper(DetectLanguage):
    """
    Class for language detection using the FasterWhisper model.

    Attributes:
        None
    """

    def execute_segment(self, speaker: str, speech_offset: float, speech_segment_float32_16khz: ndarray,
                        languages=None, **kwargs: dict) -> Transcription:
        """
        Executes language detection on a speech segment using the FasterWhisper model.

        Args:
            speaker (str): The speaker identifier.
            speech_offset (float): The offset in seconds from the beginning of the audio.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            languages (Optional[List[str]]): List of target languages for detection.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription object containing the detected language information.
        """
        transcription = Transcription()

        # Load FasterWhisper model
        model = FasterWhisperModel().model

        # Extract features from the speech segment
        features = model.feature_extractor(speech_segment_float32_16khz)

        # Limit the features to the maximum number of frames supported by the model
        segment = features[:, : model.feature_extractor.nb_max_frames]

        # Encode the segment using the FasterWhisper model
        encoder_output = model.encode(segment)

        # Detect language using the model
        results = model.model.detect_language(encoder_output)[0]

        # Parse language names to strip out markers
        all_language_probs = [(token[2:-2], prob) for (token, prob) in results]
        detected_languages = {l: p for l, p in all_language_probs}

        # Determine the best language based on user-specified languages or the maximum probability
        best_language, best_language_prob = self.determine_best_language(detected_languages, languages)

        # Create an Utterance object with the detected language information
        utterance = Utterance(
            start=speech_offset,
            end=speech_offset + len(speech_segment_float32_16khz) / 16000,
            language=best_language,
            confidence=best_language_prob,
            speaker=speaker,
            words=[])

        # Append the Utterance to the transcription
        transcription.append(utterance)

        # Log information about the detected utterance
        LOG.info(utterance.get_colour_text())

        return transcription

    def determine_best_language(self, detected_languages, target_languages):
        """
        Determines the best language based on detected languages and user-specified target languages.

        Args:
            detected_languages (dict): Dictionary of detected languages and their probabilities.
            target_languages (List[str]): List of target languages.

        Returns:
            Tuple[str, float]: The best language and its probability.
        """
        best_language = None
        best_language_prob = 0

        if target_languages and len(target_languages) > 0:
            for lang in target_languages:
                if lang in detected_languages:
                    if detected_languages[lang] > best_language_prob:
                        best_language_prob = detected_languages[lang]
                        best_language = lang

        if best_language is None:
            best_language = max(detected_languages, key=detected_languages.get)
            best_language_prob = detected_languages[best_language]

        return best_language, best_language_prob
