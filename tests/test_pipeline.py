import os
import sys

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verbatim.pipeline import Pipeline
from verbatim.context import Context


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%SZ')

    print(os.getcwd())
    context: Context = Context(source_file="tests/data/test.mp3", languages=["en", "fr", "es", "it"], nb_speakers=2)
    Pipeline(context).execute()
