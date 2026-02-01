import json
import os
from typing import List, Optional, TextIO

from verbatim.transcript.words import Utterance, Word

from .file import FileFormatter
from .writer import TranscriptWriter, TranscriptWriterConfig


class TranscriptFormatter:
    def __init__(self):
        self.first_utterance = True

    def start(self) -> bytes:
        self.first_utterance = True
        return '{\n  "utterances": [\n'.encode("utf-8")

    def finish(self) -> bytes:
        return "\n  ]\n}\n".encode("utf-8")

    def format_utterance(self, utterance: Utterance, with_words: bool = True) -> bytes:
        chunks: List[str] = []
        if not self.first_utterance:
            chunks.append(",\n")
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
                    "start": round(word.start_ts / 16000, 5),
                    "end": round(word.end_ts / 16000, 5),
                }
                for word in utterance.words
            ]

        indented_lines = "\n".join("    " + line for line in json.dumps(utterance_dict, indent=2).splitlines())

        # Use json.dumps to write the formatted JSON
        chunks.append(indented_lines)
        return "".join(chunks).encode("utf-8")


class TranscriptParser:
    def __init__(self, sample_rate: int = 16000):
        """
        :param sample_rate: The number of samples per second used to convert
                            the floating-point seconds back into integer sample counts.
        """
        self.sample_rate = sample_rate

    def parse(self, in_file: TextIO) -> List[Utterance]:
        """
        Parses a JSON transcript (following the structure produced by TranscriptFormatter)
        and returns a list of Utterance objects.

        :param in_file: A file-like object to read the JSON transcript from.
        :return: List of Utterance objects.
        """
        # Load the JSON data.
        data = json.load(in_file)

        # Retrieve the list of utterance dictionaries; default to empty list if not present.
        utt_dicts = data.get("utterances", [])
        utterances: List[Utterance] = []

        for utt in utt_dicts:
            # Retrieve basic fields.
            utterance_id = utt.get("id")
            speaker = utt.get("speaker")  # Can be None
            text = utt.get("text", "")

            # The "start" and "end" are stored in seconds (with rounding).
            start_sec = utt.get("start", 0.0)
            end_sec = utt.get("end", 0.0)
            # Convert seconds back to sample counts.
            start_ts = int(round(start_sec * self.sample_rate))
            end_ts = int(round(end_sec * self.sample_rate))

            # Process the words if they exist.
            words: List[Word] = []
            for wd in utt.get("words", []):
                if not isinstance(wd, dict):
                    continue
                # Note: the JSON key is "text" but our Word expects attribute 'word'.
                word_obj = Word(
                    word=wd.get("text", ""),
                    lang=wd.get("lang", ""),
                    probability=wd.get("prob", 0.0),
                    start_ts=int(wd.get("start", start_sec) * self.sample_rate),
                    end_ts=int(wd.get("end", end_ts) * self.sample_rate),
                )
                words.append(word_obj)

            # Create the Utterance object.
            utterance = Utterance(utterance_id=utterance_id, speaker=speaker, start_ts=start_ts, end_ts=end_ts, text=text, words=words)
            utterances.append(utterance)

        return utterances


class JsonTranscriptWriter(TranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig):
        super().__init__(config)
        self.formatter: TranscriptFormatter = TranscriptFormatter()

    def format_start(self) -> bytes:
        return self.formatter.start()

    def format_utterance(
        self,
        utterance: Utterance,
        unacknowledged_utterance: Optional[List[Utterance]] = None,
        unconfirmed_words: Optional[List[Word]] = None,
    ):
        return self.formatter.format_utterance(utterance=utterance)

    def format_end(self) -> bytes:
        return self.formatter.finish()


class JsonlTranscriptWriter(TranscriptWriter):
    def format_utterance(
        self,
        utterance: Utterance,
        unacknowledged_utterance: Optional[List[Utterance]] = None,
        unconfirmed_words: Optional[List[Word]] = None,
    ):
        utterance_dict = {
            "id": utterance.utterance_id,
            "start": round(utterance.start_ts / 16000, 5),
            "end": round(utterance.end_ts / 16000, 5),
            "speaker": utterance.speaker,
            "language": utterance.words[0].lang,
            "text": utterance.text,
            "words": [
                {
                    "text": word.word,
                    "lang": word.lang,
                    "prob": round(word.probability, 4),
                    "start": round(word.start_ts / 16000, 5),
                    "end": round(word.end_ts / 16000, 5),
                }
                for word in utterance.words
            ],
        }
        return (json.dumps(utterance_dict, ensure_ascii=False) + "\n").encode("utf-8")


def save_utterances(path: str, utterance: List[Utterance], config: Optional[TranscriptWriterConfig]):
    if config is None:
        config = TranscriptWriterConfig()
    writer: JsonTranscriptWriter = JsonTranscriptWriter(config=config)
    formatter = FileFormatter(writer=writer, output_path=os.path.splitext(path)[0] + ".json")
    formatter.open()
    for u in utterance:
        formatter.write(u)
    formatter.close()


def read_utterances(path: str) -> List[Utterance]:
    parser = TranscriptParser()
    with open(path, "r", encoding="utf-8") as file:
        utterances = parser.parse(file)
    return utterances


def read_dlm_utterances(path: str) -> List[Utterance]:
    """
    Reads utterances from a simplified JSON format that contains reference information
    but no timing data.

    Expected JSON format:
    {
      "utterances": [
        {
          "utterance_id": "utt19",
          "ref_spk": "1 1 1 1 1 1 1 1",
          "ref_text": "Also wir haben heute jetzt 30 Minuten Zeit."
        },
        ...
      ]
    }

    :param path: Path to the JSON file
    :return: List of Utterance objects with basic information (no timing data)
    """
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    utterances = []
    for utt in data.get("utterances", []):
        # Extract the speaker from ref_spk if available
        ref_spk = utt.get("ref_spk", "")
        speaker = f"SPEAKER_{ref_spk.split()[0]}" if ref_spk else None

        # Create an Utterance with minimal information
        utterance = Utterance(
            utterance_id=utt.get("utterance_id", ""),
            speaker=speaker,
            start_ts=0,  # No timing information available
            end_ts=0,  # No timing information available
            text=utt.get("ref_text", ""),
            words=[],  # No word-level information available
        )
        utterances.append(utterance)

    return utterances
