from typing import List
from .words import VerbatimUtterance

class Transcript:
    def __init__(self):
        self.utterances:List[VerbatimUtterance] = []
