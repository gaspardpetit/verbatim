import os
import logging
import soundfile as sf
import numpy as np

from verbatim.audio.convert_to_wav import ConvertToWav

LOG = logging.getLogger(__name__)

class ConvertToWavSoundfile(ConvertToWav):

    def execute(self, source_file_path: str, audio_file_path: str, **kwargs: dict):
        # Use ffmpeg from the singleton instance to convert input file to raw PCM
        output_directory = os.path.dirname(audio_file_path)
        os.makedirs(output_directory, exist_ok=True)
        data, samplerate = sf.read(source_file_path)

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        if data.dtype.kind == 'f':
            scaled_data = np.int32(data * np.iinfo(np.int32).max)
        elif data.dtype.kind == 'i':
            if data.dtype.itemsize == 4:
                scaled_data = data
            else:
                scaled_data = np.int32(data.astype(np.float32) / (2 ** (8 * data.dtype.itemsize - 1)))
        else:
            raise ValueError("Unsupported data type")

        sf.write(audio_file_path, scaled_data, samplerate, subtype='PCM_32')
