import logging
import os
from pathlib import Path

import yaml

LOG = logging.getLogger(__name__)


class Context:
    source_file_path: str
    work_directory_path: str
    audio_file_path: str
    voice_file_path: str
    diarization_file: str
    language_file: str
    transcription_path: str
    output_file: str
    languages: [str]
    nb_speakers: int

    def __init__(self, source_file: str, languages: [str] = None, nb_speakers=1):
        super().__init__()
        print(os.getcwd())
        self.source_file_path = os.path.abspath(source_file)
        self.work_directory_path = os.path.abspath("out")
        source_name, _ = os.path.splitext(os.path.basename(self.source_file_path))
        self.audio_file_path = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.audio.wav"))
        self.voice_file_path = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.voices.wav"))
        self.diarization_file = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.rttm"))
        self.language_file = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.lang.json"))
        self.transcription_path = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.script.json"))
        self.output_file = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.output"))
        self.languages = languages
        self.nb_speakers = nb_speakers

    def __str__(self):
        return yaml.dump(self)

    def to_dict(self) -> dict:
        return self.__dict__
