import os
from abc import ABC, abstractmethod
from numpy import ndarray
from pyannote.core import Annotation

from ..transcription import Transcription, Utterance

class DetectLanguage(ABC):
    """
    Abstract base class for language detection.

    Attributes:
        None
    """

    @abstractmethod
    def execute_segment(self, speaker: str, speech_offset: float, speech_segment_float32_16khz: ndarray,
                        languages=None, **kwargs: dict) -> Transcription:
        """
        Abstract method to be implemented by subclasses for executing language detection on a speech segment.

        Args:
            speaker (str): The speaker identifier.
            speech_offset (float): The offset in seconds from the beginning of the audio.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            languages (Optional[List[str]]): List of target languages for detection.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription object containing the detected language information.
        """
        ...

    @staticmethod
    def get_speech_segment(speech_segment_float32_16khz: ndarray, start: float, end: float) -> ndarray:
        """
        Extracts a subsegment from the given speech segment based on start and end time.

        Args:
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            start (float): Start time of the subsegment in seconds.
            end (float): End time of the subsegment in seconds.

        Returns:
            ndarray: Subsegment of the speech segment.
        """
        start_index = int(start * 16000)
        end_index = int(end * 16000)
        return speech_segment_float32_16khz[start_index:end_index]

    def execute(self, diarization: Annotation, speech_segment_float32_16khz: ndarray,
                language_file: str, languages=None, **kwargs: dict) -> Transcription:
        """
        Executes language detection on the entire audio based on speaker diarization.

        Args:
            diarization (Annotation): Speaker diarization information.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            language_file (str): File path to save the detected language information.
            languages (Optional[List[str]]): List of target languages for detection.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription object containing the detected language information.
        """

        transcription = Transcription()
        last_time: float = 0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            skip_time = turn.start - last_time
            if skip_time > 0.25:
                segment_transcription: Transcription = self.execute_segment(
                    speaker="none",
                    speech_offset=last_time,
                    speech_segment_float32_16khz=DetectLanguage.get_speech_segment(speech_segment_float32_16khz,
                                                                                   last_time, turn.start),
                    languages=languages)
                for utterance in segment_transcription.utterances:
                    transcription.append(utterance)
            last_time = turn.end

            segment_transcription: Transcription = self.execute_segment(
                speaker=speaker,
                speech_offset=turn.start,
                speech_segment_float32_16khz=DetectLanguage.get_speech_segment(speech_segment_float32_16khz, turn.start,
                                                                               turn.end),
                languages=languages)

            for utterance in segment_transcription.utterances:
                transcription.append(utterance)

        audio_end = len(speech_segment_float32_16khz) / 16000
        skip_time = audio_end - last_time
        if skip_time > 0.25:
            segment_transcription: Transcription = self.execute_segment(
                speaker="none",
                speech_offset=last_time,
                speech_segment_float32_16khz=DetectLanguage.get_speech_segment(speech_segment_float32_16khz, last_time,
                                                                               audio_end),
                languages=languages)
            for utterance in segment_transcription.utterances:
                transcription.append(utterance)

        """
        as a second pass, identify short segments with low confidence, and attempt to
        merge with high confidence neighbors
        """

        triplets: list[list[Utterance]] = []
        for utterance in transcription.utterances:
            if len(triplets) == 0:
                triplets.append([utterance])
            if len(triplets) == 1:
                triplets[0].append(utterance)
                triplets.append([utterance])
            triplets[len(triplets) - 2].append(utterance)
            triplets[len(triplets) - 1].append(utterance)
            triplets.append([utterance])
        triplets[len(triplets) - 2].append(utterance)
        triplets[len(triplets) - 1].append(utterance)
        triplets[len(triplets) - 1].append(utterance)

        for triplet in triplets:
            left = triplet[0]
            center = triplet[1]
            right = triplet[2]

            if center.confidence < 0.8 or center.end - center.start < 1:
                votes = {}
                votes_confidence = {}
                for pad in [0.5, 1.0, 1.5, 2.0, 2.5]:
                    start = max(0, center.start - pad)
                    end = min(len(speech_segment_float32_16khz) / 16000, center.end + pad)

                    segment_transcription: Transcription = self.execute_segment(
                        speaker=center.speaker,
                        speech_offset=start,
                        speech_segment_float32_16khz=DetectLanguage.get_speech_segment(speech_segment_float32_16khz,
                                                                                       start, end),
                        languages=languages)
                    confidence = 1
                    lang = set()
                    for utterance in segment_transcription.utterances:
                        if utterance.end >= center.start and utterance.start < center.end:
                            lang.add(utterance.language)
                            confidence = min(confidence, utterance.confidence)
                    if len(lang) == 1:
                        votes.setdefault(next(iter(lang)), 0)
                        votes[next(iter(lang))] = votes[next(iter(lang))] + 1
                        votes_confidence.setdefault(next(iter(lang)), confidence)
                        votes_confidence[next(iter(lang))] = max(confidence, votes_confidence[next(iter(lang))])
                max_votes = max(votes.values())
                max_confidence = max(v for k, v in votes_confidence.items() if votes[k] == max_votes)
                max_lang = next((k for k, v in votes_confidence.items() if v == max_confidence), None)
                if max_confidence > center.confidence:
                    if max_lang is not None and max_lang != center.language:
                        center.language = max_lang
                    center.confidence = max_confidence

        changed = True
        i = 0
        while changed:
            changed = False
            i += 1
            for triplet in triplets:
                left = triplet[0]
                center = triplet[1]
                right = triplet[2]

                print(f"[{i}] at: {center.start}")
                if center.confidence < 0.5 or center.end - center.start < 2:
                    if left.language == right.language and left.language != center.language:
                        start = left.start
                        end = right.end

                        segment_transcription: Transcription = self.execute_segment(
                            speaker=center.speaker,
                            speech_offset=start,
                            speech_segment_float32_16khz=DetectLanguage.get_speech_segment(speech_segment_float32_16khz,
                                                                                           start, end),
                            languages=languages)
                        confidence = 1
                        lang = set()
                        for utterance in segment_transcription.utterances:
                            if utterance.end >= center.start and utterance.start < center.end:
                                lang.add(utterance.language)
                                confidence = min(confidence, utterance.confidence)
                        if confidence > center.confidence:
                            if len(lang) == 1 and next(iter(lang)) != center.language:
                                center.language = next(iter(lang))
                                changed = True
                            center.confidence = confidence

                    if left.language != center.language:
                        start = left.start
                        end = center.end

                        segment_transcription: Transcription = self.execute_segment(
                            speaker=center.speaker,
                            speech_offset=start,
                            speech_segment_float32_16khz=DetectLanguage.get_speech_segment(speech_segment_float32_16khz,
                                                                                           start, end),
                            languages=languages)
                        confidence = 1
                        lang = set()
                        for utterance in segment_transcription.utterances:
                            if utterance.end >= center.start and utterance.start < center.end:
                                lang.add(utterance.language)
                                confidence = min(confidence, utterance.confidence)
                        if confidence > center.confidence:
                            if len(lang) == 1 and next(iter(lang)) != center.language:
                                center.language = next(iter(lang))
                                changed = True
                            center.confidence = confidence

                    if right.language != center.language:
                        start = center.start
                        end = right.end

                        segment_transcription: Transcription = self.execute_segment(
                            speaker=center.speaker,
                            speech_offset=start,
                            speech_segment_float32_16khz=DetectLanguage.get_speech_segment(speech_segment_float32_16khz,
                                                                                           start, end),
                            languages=languages)
                        confidence = 1
                        lang = set()
                        for utterance in segment_transcription.utterances:
                            if utterance.end >= center.start and utterance.start < center.end:
                                lang.add(utterance.language)
                                confidence = min(confidence, utterance.confidence)
                        if confidence > center.confidence:
                            if len(lang) == 1 and next(iter(lang)) != center.language:
                                center.language = next(iter(lang))
                                changed = True
                            center.confidence = confidence

        transcription.save(language_file)
        return transcription
