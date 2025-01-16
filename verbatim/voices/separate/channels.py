import logging
from typing import List, Optional
import scipy.io.wavfile
import numpy as np
from pyannote.core.annotation import Annotation
from pyannote.core.segment import Segment

from .separate import SeparationStrategy
from ...audio.sources.audiosource import AudioSource
from ...audio.sources.fileaudiosource import FileAudioSource

from ...audio.audio import wav_to_int16

# Configure logger
LOG = logging.getLogger(__name__)

class ChannelSeparation(SeparationStrategy):
    def __init__(self, **kwargs):
        super().__init__()
        LOG.info("Initializing Channel-Based Separator.")

    def separate_speakers(
        self,
        *,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_speaker_wav_prefix="",
        nb_speakers: Optional[int] = None,
        start_sample: int = 0,
        end_sample: Optional[int] = None,
    ) -> List[AudioSource]:
        """
        Separate speakers based on audio channels.

        Args:
            file_path: Path to input audio file.
            out_rttm_file: Path to output RTTM file.
            out_speaker_wav_prefix: Prefix for output WAV files.
            nb_speakers: Optional number of speakers (ignored for this implementation).

        Returns:
            Tuple of (diarization annotation, dictionary mapping speaker IDs to WAV files).
        """
        # Read the audio file
        LOG.info(f"Reading input audio file: {file_path}")
        sample_rate, audio_data = scipy.io.wavfile.read(file_path)

        # Check number of channels
        if audio_data.ndim < 2:
            LOG.info("Single channel detected. Assuming one speaker.")
            audio_data = np.expand_dims(audio_data, axis=1)

        num_channels = audio_data.shape[1]
        LOG.info(f"Detected {num_channels} channel(s) in the audio file.")

        # Process each channel
        results:List[AudioSource] = []
        for channel_idx in range(num_channels):
            speaker_label = f"SPEAKER_{channel_idx}"

            # Extract the channel data
            channel_data = audio_data[:, channel_idx]

            # Convert to int16 if necessary
            if channel_data.dtype != np.int16:
                channel_data = wav_to_int16(channel_data)

            # Generate the output file name
            file_name = (
                f"{out_speaker_wav_prefix}-{speaker_label}.wav"
                if out_speaker_wav_prefix
                else f"{speaker_label}.wav"
            )

            # Save the channel as a mono WAV file
            LOG.info(f"Saving channel {channel_idx} to file: {file_name}")
            scipy.io.wavfile.write(file_name, sample_rate, channel_data)

            # Update annotation (simplified example)
            annotation = Annotation()
            annotation[Segment(0, len(channel_data) / sample_rate)] = speaker_label
            results.append(FileAudioSource(
                file=file_name,
                diarization=annotation,
                start_sample=start_sample,
                end_sample=end_sample
            ))

            # Optionally save RTTM file
            if out_rttm_file:
                speaker_out_rttm_file = f"{out_rttm_file}-{speaker_label}.rttm"
                LOG.info(f"Saving RTTM file: {speaker_out_rttm_file}")
                with open(speaker_out_rttm_file, "w", encoding="utf-8") as rttm:
                    annotation.write_rttm(rttm)

        return results
