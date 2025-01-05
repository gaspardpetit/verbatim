import os
import sys
import unittest

from verbatim.config import Config
from verbatim.verbatim import Verbatim
from verbatim.audio.sources.audiosource import AudioSource

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestStringMethods(unittest.TestCase):
    def test_verbatim(self):
        pass

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%SZ')

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Set CUDA_VISIBLE_DEVICES to -1 to use CPU

    print(os.getcwd())
    config:Config = Config(use_cpu=True).configure_languages(["fr", "en"])
    source_stream:AudioSource = config.configure_audio_source(input_source="tests/data/init.mp3")
    verbatim:Verbatim = Verbatim(config=config)
    for acknowledged_utterances, unacknowledged_utterances, unconfirmed_words in verbatim.transcribe(source_stream):
        print(acknowledged_utterances.text)
