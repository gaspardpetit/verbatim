import sys
from ..transcription import Transcription
from .write_transcript import WriteTranscript  # Assuming WriteTranscript is in the same directory.

class WriteTranscriptStdout(WriteTranscript):
    def __init__(self, with_colours=True):
        self.with_colours = with_colours

    def execute(self, transcription_path: str, output_file: str, **kwargs: dict):
        transcript:Transcription = Transcription.load(transcription_path)
        if self.with_colours:
            text:str = transcript.get_colour_text()
        else:
            text:str = transcript.get_text()

        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()
