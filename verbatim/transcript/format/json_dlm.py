import json
import re
from typing import List, TextIO, Union, Dict

from .writer import TranscriptWriter, TranscriptWriterConfig
from ..words import Utterance, Word


class JsonDiarizationLMTranscriptWriter(TranscriptWriter):
    def __init__(self, config: TranscriptWriterConfig):
        super().__init__(config)
        self.utterances = []
        self.out: Union[None, TextIO] = None
        self.utterance_counter = 0
        self.speaker_map: Dict[str, str] = {}
        self.next_speaker_id = 1

    def _get_speaker_id(self, speaker: str) -> str:
        """Convert SPEAKER_XX format to simple numbered format (1, 2, 3, etc)"""
        if not speaker:
            return "1"  # Default speaker ID

        if speaker not in self.speaker_map:
            self.speaker_map[speaker] = str(self.next_speaker_id)
            self.next_speaker_id += 1
        return self.speaker_map[speaker]

    def open(self, path_no_ext: str):
        print(f"Opening {path_no_ext}.utt.json")
        self.out = open(f"{path_no_ext}.utt.json", "w", encoding="utf-8")
        self.utterances = []
        self.utterance_counter = 0

    def close(self):
        if self.out is not None:
            json_dict = {"utterances": self.utterances}
            json.dump(json_dict, self.out, indent=2, ensure_ascii=False)
            self.out.close()

    def write(
        self,
        utterance: Utterance,
        unacknowledged_utterance: List[Utterance] | None = None,
        unconfirmed_words: List[Word] | None = None,
    ):
        text = utterance.text.strip()
        # Convert speaker IDs
        speaker_id = self._get_speaker_id(utterance.speaker)
        # Create speaker sequence matching word count
        word_count = len(text.split())
        speaker_sequence = " ".join([speaker_id] * word_count)

        # Convert utterance to the required format
        utterance_dict = {
            "utterance_id": f"utt{self.utterance_counter}",
            "hyp_text": text,
            "hyp_spk": speaker_sequence,
            "ref_text": "",  # ground truth information needs to be added manually or by another script
            "ref_spk": "",  # see above
        }

        self.utterances.append(utterance_dict)
        self.utterance_counter += 1
