import os
import sys
import unittest

from verbatim.audio.sources import FileAudioSource
from verbatim.config import Config
from verbatim.verbatim import Verbatim

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
    config:Config = Config()
    config.source = FileAudioSource("tests/data/init.mp3")
    config.lang = ["fr", "en"]
    config.device = "cpu"
    verbatim:Verbatim = Verbatim(config=config)
    for utterance in verbatim.transcribe():
        print(utterance.text)
