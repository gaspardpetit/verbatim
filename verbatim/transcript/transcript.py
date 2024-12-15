from typing import List
from verbatim.transcript.words import VerbatimUtterance

class Transcript:
    def __init__(self):
        self.utterances:List[VerbatimUtterance] = []
