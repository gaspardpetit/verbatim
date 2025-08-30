from typing import List

from .words import Utterance


class Transcript:
    def __init__(self):
        self.utterances: List[Utterance] = []
