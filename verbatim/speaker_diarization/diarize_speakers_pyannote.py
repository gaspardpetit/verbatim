import logging
import os
import torchaudio
import torch

from pyannote.core import Annotation, Segment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from ..wav_conversion import ConvertToWav
from .diarize_speakers import DiarizeSpeakers

LOG = logging.getLogger(__name__)


class DiarizeSpeakersPyannote(DiarizeSpeakers):
    """
    Diarization implementation using PyAnnote for speaker segmentation.

    This class inherits from DiarizeSpeakers and implements diarization using PyAnnote for speaker segmentation.

    Attributes:
        None
    """

    def diarize_on_silences(self, voice_file_path: str, model_pyannote_segmentation:str,
                            huggingface_token: str, **kwargs: dict) -> Annotation:
        """
        Diarize speakers based on silences using PyAnnote.

        Args:
            audio_file (str): Path to the input audio file.
            huggingface_token (str): Hugging Face token for model access.

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        model = Model.from_pretrained(model_pyannote_segmentation, use_auth_token=huggingface_token)
        if model is None:
            LOG.error(f"Failed to retrieve model {model_pyannote_segmentation}")
            raise FileNotFoundError

        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.to(torch.device(kwargs['device']))

        hyper_parameters = {
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.0,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.5
        }
        pipeline.instantiate(hyper_parameters)
        vad: Annotation = pipeline(voice_file_path)
        vad.uri = "waveform"
        return vad

    @staticmethod
    def _load_audio(audio_path: str) -> tuple:
        """
        Loads the audio waveform and sample rate from an audio file.

        Args:
            audio_path (str): Path to the input audio file.

        Returns:
            tuple: Tuple containing waveform and sample rate.
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        return waveform, 16000

    @staticmethod
    def _get_device(**kwargs:dict) -> torch.device:
        """
        Retrieves the device (CPU or CUDA) to use for diarization.

        Returns:
            torch.device: Device to use for diarization.
        """
        device = torch.device(kwargs['device'])
        LOG.info(f"Using device: {device.type}")
        return device

    @staticmethod
    def _load_pipeline(device: torch.device, model_pyannote_diarization:str, huggingface_token: str) -> Pipeline:
        """
        Loads the diarization pipeline.

        Args:
            device (torch.device): Device to use for diarization.
            huggingface_token (str): Hugging Face token for model access.

        Returns:
            Pipeline: PyAnnote diarization pipeline.
        """
        pipeline = Pipeline.from_pretrained(
            model_pyannote_diarization,
            use_auth_token=huggingface_token
        ).to(device)
        return pipeline

    def diarize_on_speakers(self, audio_file: str, num_speakers: int,
                             huggingface_token: str, model_pyannote_diarization:str, **kwargs:dict) -> Annotation:
        """
        Diarize speakers based on speaker count using PyAnnote.

        Args:
            audio_file (str): Path to the input audio file.
            num_speakers (int): Number of speakers to consider.
            huggingface_token (str): Hugging Face token for model access.

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        device: torch.device = DiarizeSpeakersPyannote._get_device(**kwargs)
        pipeline: Pipeline = DiarizeSpeakersPyannote._load_pipeline(
            device=device,
            model_pyannote_diarization=model_pyannote_diarization,
            huggingface_token=huggingface_token)
        waveform, sample_rate = DiarizeSpeakersPyannote._load_audio(audio_file)
        with ProgressHook() as hook:
            diarization: Annotation = pipeline({
                "waveform": waveform,
                "sample_rate": sample_rate,
                "window": "whole",
                "min_duration_on": 1.0,
                "min_duration_off": 0.0,
            },
                hook=hook,
                num_speakers=num_speakers,
                min_speakers=num_speakers,
                max_speakers=num_speakers
            )

        return diarization
    def execute(self, voice_file_path: str, diarization_file: str, min_speakers: int = 1, max_speakers: int = None,
                TOKEN_HUGGINGFACE: str = None, **kwargs: dict) -> Annotation:
        """
        Execute the diarization process using PyAnnote.

        Args:
            voice_file_path (str): Path to the input audio file.
            diarization_file (str): Path to the output RTTM (Rich Transcription Time Marked) file.
            min_speakers (int, optional): Minimum number of expected speakers. Default is 1.
            max_speakers (int, optional): Maximum number of expected speakers. Default is None (unbounded).
            TOKEN_HUGGINGFACE (str, optional): Hugging Face token for customization. Default is None.
            **kwargs (dict): Additional parameters (not used in this method).

        Returns:
            Annotation: Pyannote Annotation object containing information about speaker diarization.
        """
        diarization: Annotation = None
        huggingface_token: str = TOKEN_HUGGINGFACE or os.environ.get('TOKEN_HUGGINGFACE')
        if huggingface_token is None:
            LOG.warning("No HuggingFace token was provided (TOKEN_HUGGINGFACE)")
        if min_speakers == 1 and max_speakers == 1:
            try:
                diarization = self.diarize_on_silences(voice_file_path=voice_file_path,
                                                    huggingface_token=huggingface_token, **kwargs)
            except FileNotFoundError:
                # could not log model, default on full length diarization
                LOG.warning("Failed to compute diarization, defaulting on full length audio")
                diarization  = Annotation("waveform")
                audio = ConvertToWav.load_float32_16khz_mono_audio(voice_file_path)
                diarization[Segment(0, len(audio)/16000)] = "speaker"
        else:
            diarization = self.diarize_on_speakers(voice_file_path, max_speakers, huggingface_token, **kwargs)

        with open(diarization_file, "w", encoding="utf-8") as f:
            diarization.write_rttm(f)
        LOG.info(diarization)
        return diarization
