import os
import logging
from ffmpeg import FFmpeg, Progress
from ..wav_conversion.convert_to_wav import ConvertToWav

LOG = logging.getLogger(__name__)

class FFMpegInstance:
    """
    Singleton class to manage a single instance of FFmpeg.

    This class ensures that only one instance of FFmpeg is created and used throughout the application.
    """

    __instance = None

    def __new__(cls):
        """
        Create a new instance of FFMpegInstance if one does not exist, otherwise return the existing instance.

        Returns:
            FFMpegInstance: The singleton instance of FFMpegInstance.
        """
        if cls.__instance is None:
            cls.__instance = super(FFMpegInstance, cls).__new__(cls)
            cls.__instance._init_once()
        return cls.__instance

    # pylint: disable=attribute-defined-outside-init
    def _init_once(self):
        """
        Initialize the singleton instance with an instance of FFmpeg and set up event listeners.
        """
        self.ffmpeg: FFmpeg = FFmpeg().option("y")
        self.ffmpeg.add_listener("progress", FFMpegInstance._on_progress)
        self.ffmpeg.add_listener("start", FFMpegInstance._on_start)
        self.ffmpeg.add_listener("completed", FFMpegInstance._on_completed)
        self.ffmpeg.add_listener("terminated", FFMpegInstance._on_terminated)
        self.ffmpeg.add_listener("stderr", FFMpegInstance._on_stderr)

    @staticmethod
    def _on_terminated():
        """
        Callback function called when ffmpeg is terminated.
        """
        LOG.warning("ffmpeg was terminated")

    @staticmethod
    def _on_completed():
        """
        Callback function called when ffmpeg completes successfully.
        """
        LOG.info("ffmpeg completed")

    @staticmethod
    def _on_progress(progress: Progress):
        """
        Callback function called to report the progress of ffmpeg.

        Args:
            progress (Progress): Progress object containing information about the conversion progress.
        """
        LOG.info(progress)

    @staticmethod
    def _on_start(arguments: list[str]):
        """
        Callback function called when ffmpeg starts.

        Args:
            arguments (list[str]): List of arguments used to start ffmpeg.
        """
        LOG.info(f"ffmpeg started: {' '.join(arguments)}")

    @staticmethod
    def _on_stderr(text: str):
        """
        Callback function called when there is an error message in ffmpeg's stderr.

        Args:
            text (str): Error message from ffmpeg's stderr.
        """
        LOG.debug(text)


class ConvertToWavFFMpeg(ConvertToWav):

    def execute(self, source_file_path: str, audio_file_path: str, **kwargs: dict):
        """
        Convert audio file to WAV format using ffmpeg.

        Args:
            source_file_path (str): Path to the input audio file.
            audio_file_path (str): Path to save the output WAV file.
            **kwargs (dict): Additional parameters (not used in this method).
        """
        LOG.info(f"Converting {source_file_path} to {audio_file_path}")

        # Use ffmpeg from the singleton instance to convert input file to raw PCM
        output_directory = os.path.dirname(audio_file_path)
        os.makedirs(output_directory, exist_ok=True)

        ffmpeg = FFMpegInstance().ffmpeg
        ffmpeg_command = ffmpeg.input(source_file_path).output(audio_file_path, ac='1', acodec='pcm_s32le')
        ffmpeg_command.execute()
