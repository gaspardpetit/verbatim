import logging
import os
import wave
from typing import Dict

import numpy as np
from pyannote.core.annotation import Annotation

from .audiosource import AudioSource
from verbatim.audio.audio import FormatAudio
from verbatim.audio.audio import convert_mp3_to_wav
from verbatim.voices.diarization import Diarization
from verbatim.voices.isolation import VoiceSeparator

LOG = logging.getLogger(__name__)


class FileAudioSource(AudioSource):
    diarization:Annotation
    speaker_audio:Dict[str, np.array]
    stream:wave.Wave_read = None

    def __init__(self, file: str):
        super().__init__()
        self.file_path = file
        if self.file_path.endswith(".mp3"):
            # Convert mp3 to wav
            wav_file_path = self.file_path.replace(".mp3", ".wav")
            convert_mp3_to_wav(self.file_path, wav_file_path)
            self.file_path = wav_file_path
    
    def compute_diarization(self, device:str, rttm_file:str = None, nb_speakers:int = None) -> Annotation:
        diarization = None
        try:
            diarization = Diarization(device=device, huggingface_token=os.getenv("HUGGINGFACE_TOKEN"))
            annotation = diarization.compute_diarization(file_path=self.file_path, out_rttm_file=rttm_file, nb_speakers=nb_speakers)
            return annotation
        finally:
            if diarization:
                del diarization

    def isolate_voices(self, out_path_prefix:str=None):
        LOG.info("Initializing Voice Isolation Model.")
        voice_separator:VoiceSeparator = None
        try:
            voice_separator = VoiceSeparator(log_level=LOG.level)
            if not out_path_prefix:
                basename, _ = os.path.splitext(os.path.basename(self.file_path))
                voice_prefix = f"{basename}-voice"
                noise_prefix = f"{basename}-noise"
            else:
                basename, ext = os.path.splitext(out_path_prefix)
                if ext:
                    voice_prefix = basename
                    noise_prefix = f"{basename}-noise"
                else:
                    voice_prefix = f"{basename}-voice"
                    noise_prefix = f"{basename}-noise"

            self.file_path = voice_separator.isolate_voice_in_file(
                file=self.file_path,
                out_voice=voice_prefix,
                out_noise=noise_prefix,
                )
        finally:
            if voice_separator:
                del voice_separator
            
    def next_chunk(self, chunk_length=1) -> np.ndarray:
        LOG.info(f"Reading {chunk_length} seconds of audio from file.")
        frames = self.stream.readframes(int(self.stream.getframerate() * chunk_length))
        sample_width = self.stream.getsampwidth()
        n_channels = self.stream.getnchannels()
        dtype = np.int16 if sample_width == 2 else np.int32 if sample_width == 4 else np.uint8
        audio_array = np.frombuffer(frames, dtype=dtype)
        audio_array = audio_array.reshape(-1, n_channels)

        if len(audio_array) == 0:
            return audio_array
        
        audio_array = FormatAudio(audio_array, from_sampling_rate=self.stream.getframerate())
        
        LOG.info("Finished reading audio chunk from file.")
        return audio_array

    def has_more(self):
        current_frame = self.stream.tell()
        total_frames = self.stream.getnframes()
        return current_frame < total_frames

    def open(self):
        self.stream = wave.open(self.file_path, 'rb')

    def close(self):
        self.stream.close()