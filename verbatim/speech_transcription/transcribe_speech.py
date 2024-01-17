import os
import logging
from abc import abstractmethod
import numpy as np
from numpy import ndarray
from pyannote.core import Annotation
from pyannote.database.util import load_rttm

from ..transcription import Transcription
from ..speaker_diarization import DiarizeSpeakersSpeechBrain
from ..wav_conversion import ConvertToWav
from ..filter import Filter

LOG = logging.getLogger(__name__)


class TranscribeSpeech(Filter):
    """
    Abstract class for transcribing audio.

    Attributes:
        None
    """

    @abstractmethod
    def execute_segment(self, speech_segment_float32_16khz: ndarray,
                        speaker: str = "speaker", speech_offset: float = 0,
                        language: str = None, prompt: str = "",
                        **kwargs: dict) -> Transcription:
        """
        Abstract method for executing transcription on a speech segment.

        Args:
            speaker (str): The speaker identifier.
            speech_offset (float): The offset in seconds from the beginning of the audio.
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            language (str): Target language for transcription.
            prompt (str): Optional transcription prompt.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription object containing the transcribed information.
        """


    def _optimize_sequence(self, sequence: dict, full_duration: float) -> dict:
        """
        Optimizes the sequence by ensuring language and speaker information is set consistently.

        Args:
            sequence (dict): Dictionary containing language and speaker information at different time points.
            full_duration (float): Full duration of the audio in seconds.

        Returns:
            dict: Optimized sequence.
        """
        # ensure that language is set from the start
        sequence = dict(sorted(sequence.items()))
        if int(0) not in sequence.keys():
            sequence[0] = sequence[sequence.keys()[0]]

        # ensure that language is set until the end
        last_index: int = int(full_duration * 16000)
        sequence = dict(sorted(sequence.items()))
        if last_index not in sequence:
            sequence[last_index] = sequence[next(iter(reversed(sequence.keys())))]

        # propagate language over short gaps
        changed2 = True
        while changed2:
            changed2 = False
            changed = True
            while changed:
                changed = False
                index_seq = sorted(sequence.items())
                _, first = index_seq[0]
                _, after = index_seq[1]
                if first['language'] == "none" and after['language'] != "none":
                    first['language'] = after['language']
                    changed = True
                if first['speaker'] == "none" and after['speaker'] != "none":
                    first['speaker'] = after['speaker']
                    changed = True

                _, last = index_seq[len(index_seq) - 1]
                _, prev = index_seq[len(index_seq) - 2]
                if last['language'] == "none" and prev['language'] != "none":
                    last['language'] = prev['language']
                    changed = True
                if last['speaker'] == "none" and prev['speaker'] != "none":
                    last['speaker'] = prev['speaker']
                    changed = True

                for i in range(1, len(index_seq) - 1):
                    _, prev = index_seq[i - 1]
                    _, after = index_seq[i + 1]
                    _, cur = index_seq[i]

                    if cur['language'] == "none":
                        if prev['language'] == after['language'] and cur['language'] != prev['language']:
                            cur['language'] = prev['language']
                            changed = True

                        elif prev['language'] != "none":
                            cur['language'] = prev['language']
                            changed = True

                        elif after['language'] != "none":
                            cur['language'] = after['language']
                            changed = True

                    if cur['speaker'] == "none":
                        if prev['speaker'] == after['speaker'] and cur['speaker'] != prev['speaker']:
                            cur['speaker'] = prev['speaker']
                            changed = True
                        elif prev['speaker'] != "none":
                            cur['speaker'] = prev['speaker']
                            changed = True

                        elif after['speaker'] != "none":
                            cur['speaker'] = after['speaker']
                            changed = True

                min_duration = 1
                current_start = 0
                current_state = None
                for time, state in sorted(sequence.items()):
                    if current_state is None:
                        current_start = time
                        current_state = state
                    else:
                        duration = (time - current_start) / 16000
                        if duration < min_duration:
                            if current_state['language'] == "none" and state['language'] != "none":
                                current_state['language'] = state['language']
                                changed = True
                                changed2 = True
                            if current_state['speaker'] == "none" and state['speaker'] != "none":
                                current_state['speaker'] = state['speaker']
                                changed = True
                                changed2 = True
                        current_start = time
                        current_state = state

            changed = True
            while changed:
                changed = False
                min_duration = 1
                current_start = 0
                current_state = None
                for time, state in reversed(sorted(sequence.items())):
                    if current_state is None:
                        current_start = time
                        current_state = state
                    else:
                        duration = -(time - current_start) / 16000
                        if duration < min_duration:
                            if current_state['language'] == "none" and state['language'] != "none":
                                current_state['language'] = state['language']
                                changed = True
                                changed2 = True
                            if current_state['speaker'] == "none" and state['speaker'] != "none":
                                current_state['speaker'] = state['speaker']
                                changed = True
                                changed2 = True
                        current_start = time
                        current_state = state

        return sequence

    def _prepare_second_pass_diarization(self, diarization: Annotation, detected_languages: Transcription) -> dict:
        """
        Prepares the diarization for the second pass by creating a sparse sequence.

        Args:
            diarization (Annotation): Speaker diarization information.
            detected_languages (Transcription): Transcription containing detected language information.

        Returns:
            dict: Sparse sequence for the second pass.
        """

        sparse_sequence = {}
        sparse_sequence[0] = {"speaker": "none", "language": "none"}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            sparse_sequence.setdefault(int(turn.start * 16000), {}).update(
                {"speaker": speaker})
            sparse_sequence.setdefault(int(turn.end * 16000), {}).update(
                {"speaker": "none"})

        for utterance in detected_languages.utterances:
            sparse_sequence.setdefault(int(utterance.start * 16000), {}).update(
                {"language": utterance.language})
            sparse_sequence.setdefault(int(utterance.end * 16000), {}).update(
                {"language": "none"})

        state = {"language": "none", "speaker": "none"}
        sequence = {0: state}
        for index, sparse_state in sorted(sparse_sequence.items()):
            state = dict(state)
            state.update(sparse_state)
            sequence[int(index)] = state

        return sequence

    @staticmethod
    def apply_cosine_fade(samples: ndarray, fade_start: int, fade_end: int, start: int, end: int):
        """
        Applies a cosine fade to a portion of audio.

        Args:
            samples (ndarray): Audio samples.
            fade_start (int): Start index of the fade.
            fade_end (int): End index of the fade.
            start (int): Start index of the audio portion.
            end (int): End index of the audio portion.
        """
        if fade_start > 0:
            fade_out = np.cos(np.linspace(0, np.pi / 2, fade_start))
            samples[start:start + fade_start] *= fade_out

        if fade_end > 0:
            fade_in = np.cos(np.linspace(np.pi / 2, 0, fade_end))
            samples[end - fade_end:end] *= fade_in

        # Mute the audio in between fade-in and fade-out
        samples[start + fade_start:end - fade_end] = 0.0

    def execute_for_speaker_and_language(self, speech_segment_float32_16khz, sequence,
                                         speaker: str, language: str, **kwargs: dict):
        """
        Executes transcription for a specific speaker and language.

        Args:
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            sequence (dict): Dictionary containing language and speaker information at different time points.
            speaker (str): The speaker identifier.
            language (str): Target language for transcription.

        Returns:
            Transcription: Transcription object containing the transcribed information.
        """
        audio_lang = speech_segment_float32_16khz.copy()

        mute_seq = {}
        mute_start = None
        mute_end = None
        for index, info in sequence.items():
            if info['speaker'] != speaker or (
                    (language is None or info['language'] != language) and info['language'] != "none"):
                if mute_start is None:
                    mute_start = index
                mute_end = index
            else:
                if mute_start is not None:
                    mute_end = index
                    mute_seq[mute_start] = mute_end
                    mute_start = None
                    mute_end = None
        if mute_start is not None:
            mute_seq[mute_start] = len(audio_lang)

        for mute_start, mute_end in mute_seq.items():
            fade_duration = 0.1 * 16000  # Adjust this parameter for the fade duration
            fade_start = min(len(audio_lang) - mute_start, int(max(0, min(mute_start, fade_duration / 2))) * 2)
            fade_end = int(max(0, min(len(audio_lang) - mute_end, fade_duration / 2))) * 2
            TranscribeSpeech.apply_cosine_fade(
                samples=audio_lang,
                fade_start=int(fade_start),
                fade_end=int(fade_end),
                start=int(mute_start - fade_start / 2),
                end=int(mute_end + fade_end / 2),
            )

        # speechbrain throws an exception when processing audio shorter than 30s
        # https://github.com/speechbrain/speechbrain/issues/2334
        audio_lang = DiarizeSpeakersSpeechBrain.pad_audio_to_duration(audio_lang, 31, 16000)

        work_directory_path = kwargs['work_directory_path'] or "."
        speaker_lang_audio_path = os.path.join(work_directory_path, f"{speaker}-{language}.wav")
        ConvertToWav.save_float32_16khz_mono_audio(audio_lang, speaker_lang_audio_path)

        segments = DiarizeSpeakersSpeechBrain().diarize_on_silences(
            audio_path=speaker_lang_audio_path, **kwargs)
        prompt = "Hello and welcome. This is my presentation."
        whole_transcription = Transcription()
        for turn, _, _ in segments.itertracks(yield_label=True):
            sample_start = int(turn.start * 16000)
            sample_end = int(turn.end * 16000)
            transcription: Transcription = self.execute_segment(
                speaker=speaker,
                speech_offset=turn.start,
                speech_segment_float32_16khz=audio_lang[sample_start:sample_end],
                prompt=prompt,
                language=language,
                **kwargs)
            prompt = transcription.get_text()
            for u in transcription.utterances:
                whole_transcription.append(u)
        return whole_transcription

    # pylint: disable=unused-argument
    # pylint: disable=arguments-differ
    def execute(self, voice_file_path:str, language_file:str,
                transcription_path: str, diarization_file:str, languages: list, **kwargs: dict) -> Transcription:
        """
        Executes the transcription process for multiple speakers and languages.

        Args:
            speech_segment_float32_16khz (ndarray): Speech segment data in float32 format at 16kHz.
            detected_languages (Transcription): Transcription containing detected language information.
            transcription_path (str): File path to save the final transcription result.
            diarization (Annotation): Speaker diarization information.
            languages (list): List of target languages.
            **kwargs (dict): Additional keyword arguments for customization.

        Returns:
            Transcription: Transcription object containing the final result.
        """

        speech_segment_float32_16khz = ConvertToWav.load_float32_16khz_mono_audio(voice_file_path)
        detected_languages: Transcription = Transcription.load(language_file)
        rttms = load_rttm(diarization_file)
        diarization = next(iter(rttms.values()))

        full_transcription = Transcription()
        sequence = self._prepare_second_pass_diarization(diarization=diarization, detected_languages=detected_languages)
        sequence = self._optimize_sequence(sequence=sequence, full_duration=len(speech_segment_float32_16khz) / 16000)

        speakers = set()
        for info in sequence.values():
            speakers.add(info['speaker'])

        if languages is None or len(languages) == 0:
            for speaker in speakers:
                transcription: Transcription = self.execute_for_speaker_and_language(
                    speech_segment_float32_16khz=speech_segment_float32_16khz, sequence=sequence, speaker=speaker,
                    language=None,
                    **kwargs)
                for utterance in transcription.utterances:
                    full_transcription.append(utterance)
        else:
            for language in languages:
                for speaker in speakers:
                    transcription: Transcription = self.execute_for_speaker_and_language(
                        speech_segment_float32_16khz=speech_segment_float32_16khz, sequence=sequence, speaker=speaker,
                        language=language,
                        **kwargs)
                    for utterance in transcription.utterances:
                        full_transcription.append(utterance)

        full_transcription = full_transcription.regroup_by_words()
        full_transcription.save(transcription_path)
        LOG.info(full_transcription.get_colour_text())
        return full_transcription
