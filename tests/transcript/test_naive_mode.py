import unittest

import numpy as np

from verbatim.config import Config
from verbatim.transcript.words import Word
from verbatim.verbatim import Verbatim


class DummyTranscriber:
    def __init__(self, words, guessed_lang="en"):
        self.words = words
        self.guessed_lang = guessed_lang
        self.calls = []
        self.guess_calls = []

    def guess_language(self, audio, lang):
        self.guess_calls.append({"samples": len(audio), "lang": list(lang)})
        return self.guessed_lang, 0.99

    def transcribe(
        self,
        *,
        audio,
        lang,
        prompt,
        prefix,
        window_ts,
        audio_ts,
        whisper_beam_size,
        whisper_best_of,
        whisper_patience,
        whisper_temperatures,
    ):
        self.calls.append(
            {
                "samples": len(audio),
                "lang": lang,
                "prompt": prompt,
                "prefix": prefix,
                "window_ts": window_ts,
                "audio_ts": audio_ts,
            }
        )
        _ = whisper_beam_size, whisper_best_of, whisper_patience, whisper_temperatures
        return list(self.words)


class DummySentenceTokenizer:
    def split(self, words):
        _ = words
        return ["Hello. ", "Bye. "]


class DummyModels:
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.sentence_tokenizer = DummySentenceTokenizer()


class DummyAudioStream:
    def __init__(self, chunks, start_offset=0):
        self._chunks = list(chunks)
        self.start_offset = start_offset
        self.diarization = None
        self.total_samples = start_offset + sum(len(chunk) for chunk in chunks)
        self.end_sample = None

    def has_more(self):
        return len(self._chunks) > 0

    def next_chunk(self, chunk_length=1):
        _ = chunk_length
        return self._chunks.pop(0)

    def close(self):
        return None

    def get_nchannels(self):
        return 1

    def get_rate(self):
        return 16000


class TestNaiveMode(unittest.TestCase):
    def test_transcribe_naive_reads_full_audio_and_emits_sentence_utterances(self):
        config = Config(device="cpu")
        config.code_switching = False
        config.lang = ["en", "fr"]

        words = [
            Word(start_ts=0, end_ts=8000, word="Hello. ", probability=1.0, lang="en"),
            Word(start_ts=8000, end_ts=16000, word="Bye. ", probability=1.0, lang="en"),
        ]
        transcriber = DummyTranscriber(words=words, guessed_lang="en")
        verbatim = Verbatim(config=config, models=DummyModels(transcriber))
        audio_stream = DummyAudioStream(
            chunks=[
                np.zeros(16000, dtype=np.float32),
                np.zeros(8000, dtype=np.float32),
            ]
        )

        emitted = list(verbatim.transcribe(audio_stream=audio_stream))

        self.assertEqual(1, len(transcriber.guess_calls))
        self.assertEqual(1, len(transcriber.calls))
        self.assertEqual(24000, transcriber.calls[0]["samples"])
        self.assertEqual("en", transcriber.calls[0]["lang"])
        self.assertEqual("", transcriber.calls[0]["prefix"])
        self.assertEqual(2, len(emitted))
        self.assertEqual("Hello. ", emitted[0][0].text)
        self.assertEqual("Bye. ", emitted[1][0].text)
        self.assertEqual(["Bye. "], [utterance.text for utterance in emitted[0][1]])
        self.assertEqual([], emitted[1][1])
        self.assertEqual([], emitted[0][2])
        self.assertEqual([], emitted[1][2])


if __name__ == "__main__":
    unittest.main()
