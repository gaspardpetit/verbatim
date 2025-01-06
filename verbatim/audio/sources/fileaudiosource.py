import logging
import os
import wave
from typing import Dict, Union

import numpy as np
from pyannote.core.annotation import Annotation

from .audiosource import AudioSource, AudioStream
from ..audio import format_audio
from ..audio import convert_mp3_to_wav
from ...voices.diarization import Diarization
from ...voices.isolation import VoiceSeparator

LOG = logging.getLogger(__name__)

class FileAudioStream(AudioStream):
    source:"FileAudioSource"

    def __init__(self, source:"FileAudioSource"):
        super().__init__(start_offset=source.start_sample)
        self.source = source
        self.stream = wave.open(self.source.file_path, "rb")
        if self.source.start_sample != 0:
            self.setpos(self.source.start_sample)

    def setpos(self, new_sample_pos: int):
        file_samplerate = self.stream.getframerate()
        if file_samplerate != 16000:
            file_sample_pos = new_sample_pos * file_samplerate // 16000
        else:
            file_sample_pos = new_sample_pos
        self.stream.setpos(int(file_sample_pos))

    def next_chunk(self, chunk_length=1) -> np.ndarray:
        LOG.info(f"Reading {chunk_length} seconds of audio from file.")
        frames = self.stream.readframes(int(self.stream.getframerate() * chunk_length))
        sample_width = self.stream.getsampwidth()
        n_channels = self.stream.getnchannels()
        dtype = (
            np.int16
            if sample_width == 2
            else np.int32
            if sample_width == 4
            else np.uint8
        )
        audio_array = np.frombuffer(frames, dtype=dtype)
        audio_array = audio_array.reshape(-1, n_channels)

        if len(audio_array) == 0:
            return audio_array

        audio_array = format_audio(
            audio_array, from_sampling_rate=self.stream.getframerate()
        )

        LOG.info("Finished reading audio chunk from file.")
        return audio_array

    def has_more(self):
        current_frame = self.stream.tell()
        if self.source.end_sample is not None and current_frame > self.source.end_sample:
            return False
        total_frames = self.stream.getnframes()
        return current_frame < total_frames

    def close(self):
        self.stream.close()

class FileAudioSource(AudioSource):
    diarization: Annotation
    speaker_audio: Dict[str, np.array]
    stream: wave.Wave_read = None

    def __init__(self, file: str, start_sample: int = 0, end_sample: Union[None, int] = None):
        super().__init__(source_name=file)
        self.file_path = file
        if self.file_path.endswith(".mp3"):
            # Convert mp3 to wav
            wav_file_path = self.file_path.replace(".mp3", ".wav")
            convert_mp3_to_wav(self.file_path, wav_file_path)
            self.file_path = wav_file_path
        self.end_sample = end_sample
        self.start_sample = start_sample

    def compute_diarization(self, device: str, rttm_file: str = None, nb_speakers: int = None) -> Annotation:
        diarization = None
        try:
            diarization = Diarization(
                device=device, huggingface_token=os.getenv("HUGGINGFACE_TOKEN")
            )
            if nb_speakers == 0:
                nb_speakers = None
            annotation = diarization.compute_diarization(
                file_path=self.file_path,
                out_rttm_file=rttm_file,
                nb_speakers=nb_speakers,
            )
            return annotation
        finally:
            if diarization:
                del diarization

    def isolate_voices(self, out_path_prefix: str = None):
        LOG.info("Initializing Voice Isolation Model.")
        voice_separator: Union[None, VoiceSeparator] = None
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

            self.file_path, _ = voice_separator.isolate_voice_in_file(
                file=self.file_path,
                out_voice=voice_prefix,
                out_noise=noise_prefix,
            )
        finally:
            if voice_separator:
                del voice_separator

    def open(self):
        return FileAudioStream(source=self)
