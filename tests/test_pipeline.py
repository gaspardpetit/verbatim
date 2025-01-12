import os
import sys
import unittest

from verbatim.config import Config
from verbatim.verbatim import Verbatim
from verbatim.audio.sources.audiosource import AudioSource
from verbatim.audio.sources.factory import create_audio_source

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class TestStringMethods(unittest.TestCase):
    def test_verbatim(self):
        pass


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set CUDA_VISIBLE_DEVICES to -1 to use CPU

    print(os.getcwd())

    config: Config = Config(device="cpu").configure_languages(["fr", "en"])
    audio_source: AudioSource = create_audio_source(input_source="tests/data/init.mp3", device=config.device)
    verbatim: Verbatim = Verbatim(config=config)
    with audio_source.open() as audio_stream:
        for utterance, unack_utterances, unconfirmed_words in verbatim.transcribe(audio_stream=audio_stream):
            print(utterance.text)
