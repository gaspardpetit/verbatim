import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# pylint: disable=(wrong-import-position
from verbatim.pipeline import Pipeline
from verbatim.context import Context

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
    #context: Context = Context(source_file="tests/data/test.mp3", languages=["en", "fr", "es", "it"], nb_speakers=2)
    context: Context = Context(
        source_file="tests/data/init.mp3", languages=["en"], nb_speakers=2, log_level=logging.DEBUG, device="cpu")
    Pipeline(context).execute()
