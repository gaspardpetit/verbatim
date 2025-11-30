import logging
import os
from typing import List, Optional

import numpy as np
import scipy.io.wavfile

from verbatim.audio.audio import wav_to_int16
from verbatim.audio.sources.audiosource import AudioSource
from verbatim.audio.sources.fileaudiosource import FileAudioSource
from verbatim_diarization.separate.base import SeparationStrategy
from verbatim_rttm import Annotation, AudioRef, Segment, write_vttm

# Configure logger
LOG = logging.getLogger(__name__)


class ChannelSeparation(SeparationStrategy):
    def __init__(self, **kwargs):  # pylint: disable=unused-argument
        super().__init__()
        LOG.info("Initializing Channel-Based Separator.")

    def separate_speakers(
        self,
        *,
        file_path: str,
        out_rttm_file: Optional[str] = None,
        out_vttm_file: Optional[str] = None,
        out_speaker_wav_prefix="",
        nb_speakers: Optional[int] = None,  # pylint: disable=unused-argument
        start_sample: int = 0,
        end_sample: Optional[int] = None,
    ) -> List[AudioSource]:
        """
        Separate speakers based on audio channels.

        Args:
            file_path: Path to input audio file.
            out_rttm_file: Optional legacy RTTM output path.
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
        if audio_data.shape[1] < 1:  # type: ignore[index]
            raise ValueError("Audio data must have at least one channel")

        num_channels = int(audio_data.shape[1])  # type: ignore[index]
        LOG.info(f"Detected {num_channels} channel(s) in the audio file.")

        # Process each channel
        results: List[AudioSource] = []
        segments = []
        audio_refs = []
        for channel_idx in range(num_channels):
            speaker_label = f"SPEAKER_{channel_idx}"

            # Extract the channel data
            channel_data = audio_data[:, channel_idx]

            # Convert to int16 if necessary
            if channel_data.dtype != np.int16:
                channel_data = wav_to_int16(channel_data)

            # Generate the output file name
            file_name = f"{out_speaker_wav_prefix}-{speaker_label}.wav" if out_speaker_wav_prefix else f"{speaker_label}.wav"

            # Save the channel as a mono WAV file
            LOG.info(f"Saving channel {channel_idx} to file: {file_name}")
            scipy.io.wavfile.write(file_name, sample_rate, channel_data)

            # Update annotation for this channel
            file_id = os.path.splitext(os.path.basename(file_name))[0]
            segment = Segment(start=0.0, end=len(channel_data) / sample_rate, speaker=speaker_label, file_id=file_id)
            segments.append(segment)
            audio_refs.append(AudioRef(id=file_id, path=file_name, channel=str(channel_idx)))
            ann = Annotation([segment])
            results.append(FileAudioSource(file=file_name, diarization=ann, start_sample=start_sample, end_sample=end_sample))

        # Optionally save combined RTTM/VTTM
        if (out_rttm_file or out_vttm_file) and segments:
            combined = Annotation(segments=segments)
            if out_rttm_file:
                LOG.info("Saving RTTM file: %s", out_rttm_file)
                with open(out_rttm_file, "w", encoding="utf-8") as rttm:
                    combined.write_rttm(rttm)
            if out_vttm_file:
                LOG.info("Saving VTTM file: %s", out_vttm_file)
                write_vttm(out_vttm_file, audio=audio_refs, annotation=combined)

        return results
