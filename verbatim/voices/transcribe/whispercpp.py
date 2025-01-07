# PS08_verbatim/verbatim/voices/transcribe/whispercpp.py

import logging
from typing import List, Tuple, Union
from numpy.typing import NDArray

from pywhispercpp.model import Model

from verbatim.audio.audio import samples_to_seconds

from ...transcript.words import Word
from .transcribe import Transcriber

LOG = logging.getLogger(__name__)


class WhisperCppTranscriber(Transcriber):
    def __init__(
        self,
        *,
        model_size_or_path: str,
        device: str,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Union[None,List[float]] = None,
    ):
        if whisper_temperatures is None:
            whisper_temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        if device == "cpu":
            self.whisper_model = Model(model=model_size_or_path)
        else:
            # WhisperCPP doesn't support GPU
            LOG.warning("WhisperCPP only supports CPU inference, ignoring GPU device")
            self.whisper_model = Model(model=model_size_or_path)

        self.whisper_beam_size = whisper_beam_size
        self.whisper_best_of = whisper_best_of
        self.whisper_patience = whisper_patience
        self.whisper_temperatures = whisper_temperatures

    def guess_language(self, audio: NDArray, lang: List[str]) -> Tuple[str, float]:
        # TODO: Implement proper language detection
        # For now return first language
        return lang[0], 1.0

    def transcribe(
        self,
        *,
        audio: NDArray,
        lang: str,
        prompt: str,
        prefix: str,
        window_ts: int,
        audio_ts: int,
        whisper_beam_size: int = 3,
        whisper_best_of: int = 3,
        whisper_patience: float = 1.0,
        whisper_temperatures: Union[None,List[float]] = None,
    ) -> List[Word]:
        LOG.info(f"Transcription Prefix: {prefix}")

        # Run inference
        words = self.whisper_model.transcribe(
            audio,
            max_len=1,
            split_on_word=True,
            token_timestamps=True,
        )

        transcript_words: List[Word] = []
        for w in words:
            print("XXXX", w.text)
            # WhisperCPP segments are in centiseconds (1/100 second)
            start_ts = int(w.t0 * 160) + window_ts  # Convert to samples
            end_ts = int(w.t1 * 160) + window_ts

            if end_ts > audio_ts:
                continue

            word = Word.from_whisper_cpp_1w_segment(segment=w, lang=lang)

            LOG.debug(f"[{start_ts} ({samples_to_seconds(start_ts)})]: {word.word}")
            transcript_words.append(word)

        print(transcript_words)
        return transcript_words
