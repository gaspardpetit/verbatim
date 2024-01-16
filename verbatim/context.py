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
    log_level: int
    device: str
    model_fasterwhisper:str
    model_whisper:str
    model_pyannote_segmentation:str
    model_pyannote_diarization:str
    model_mdx:str
    model_speechbrain_vad:str

    def __init__(self, source_file: str, out_dir: str = "out", languages: [str] = None,
                  nb_speakers=1, log_level=logging.WARNING, device:str="cuda",
                  model_fasterwhisper="large-v3", model_whisper="large-v3",
                  model_pyannote_segmentation:str = "pyannote/segmentation-3.0",
                  model_pyannote_diarization:str = "pyannote/speaker-diarization-3.1",
                  model_mdx:str = 'Kim_Vocal_2', model_speechbrain_vad:str = "speechbrain/vad-crdnn-libriparty"):
        super().__init__()
        self.source_file_path = os.path.abspath(source_file)
        self.work_directory_path = os.path.abspath(out_dir)
        source_name, _ = os.path.splitext(os.path.basename(self.source_file_path))
        self.audio_file_path = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.audio.wav"))
        self.voice_file_path = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.voices.wav"))
        self.diarization_file = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.rttm"))
        self.language_file = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.lang.json"))
        self.transcription_path = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.script.json"))
        self.output_file = str(Path.joinpath(Path(self.work_directory_path), f"{source_name}.output"))
        self.languages = languages
        self.min_speakers = nb_speakers
        self.max_speakers = nb_speakers
        self.log_level = log_level
        self.device = device
        self.model_fasterwhisper = model_fasterwhisper
        self.model_whisper = model_whisper
        self.model_pyannote_segmentation = model_pyannote_segmentation
        self.model_pyannote_diarization = model_pyannote_diarization

        self.model_mdx = model_mdx
        self.model_speechbrain_vad = model_speechbrain_vad

    def __str__(self):
        return self.to_yaml()

    def to_yaml(self):
        return yaml.dump(self.to_dict())

    @staticmethod
    def from_yaml(source_file:str, out_dir:str, text:str):
        dict_data = yaml.safe_load(text)
        return Context.from_dict(source_file=source_file, out_dir=out_dir, dict_data=dict_data)

    @staticmethod
    def from_dict(source_file:str, out_dir:str, dict_data:dict):
        obj:Context = Context(source_file=source_file, out_dir=out_dir)
        obj.__dict__.update(dict_data)
        return obj

    def to_dict(self) -> dict:
        return self.__dict__
