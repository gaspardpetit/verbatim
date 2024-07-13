from typing import List
from .write_transcript import WriteTranscript

class WriteTranscriptMulti(WriteTranscript):
    def __init__(self, writers:List[WriteTranscript]):
        self.writers = writers

    def execute(self, transcription_path: str, output_file: str, **kwargs: dict):
        for writer in self.writers:
            writer.execute(transcription_path=transcription_path, output_file=output_file, **kwargs)
