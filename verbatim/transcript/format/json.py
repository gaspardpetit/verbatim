import json
from typing import TextIO, List, Optional

from .writer import TranscriptWriter, TranscriptWriterConfig
from ..words import Utterance, Word


class TranscriptFormatter:
    def __init__(self):
        self.first_utterance = True

    def open(self, out: TextIO):
        out.write('{\n  "utterances": [\n')

    def close(self, out: TextIO):
        out.write("\n  ]\n}\n")

    def format_utterance(self, utterance: Utterance, out: TextIO, with_words: bool = True):
        if not self.first_utterance:
            out.write(",\n")
        else:
            self.first_utterance = False

        # Create a dictionary for the utterance
        utterance_dict = {
            "id": utterance.utterance_id,
            "start": round(utterance.start_ts / 16000, 5),
            "end": round(utterance.end_ts / 16000, 5),
            "speaker": utterance.speaker,
            "language": utterance.words[0].lang,
            "text": utterance.text,
        }

        if with_words:
            utterance_dict["words"] = [
                {
                    "text": word.word,
                    "lang": word.lang,
                    "prob": round(word.probability, 4),
                }
                for word in utterance.words
            ]

        indented_lines = "\n".join("    " + line for line in json.dumps(utterance_dict, indent=2).splitlines())

        # Use json.dumps to write the formatted JSON
        out.write(indented_lines)


class JsonTranscriptWriter(TranscriptWriter):
    out: TextIO

    def __init__(self, config: TranscriptWriterConfig):
        super().__init__(config)
        self.formatter: TranscriptFormatter = TranscriptFormatter()

    def open(self, path_no_ext: str):
        # Open the output file
        # pyline: disable=consider-using-with
        self.out = open(f"{path_no_ext}.json", "w", encoding="utf-8")
        self.formatter.open(out=self.out)

    def close(self):
        self.formatter.close(out=self.out)
        self.out.close()

    def write(
        self,
        utterance: Utterance,
        unacknowledged_utterance: Optional[List[Utterance]] = None,
        unconfirmed_words: Optional[List[Word]] = None,
    ):
        self.formatter.format_utterance(utterance=utterance, out=self.out)
        self.out.flush()
