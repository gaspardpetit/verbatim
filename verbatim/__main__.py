import argparse
import logging
import os
import sys
import re

import numpy as np
import torch

from .__init__ import __version__
from .config import Config
from .transcript.format.writer import (
    TranscriptWriter,
    SpeakerStyle,
    TimestampStyle,
    ProbabilityStyle,
    LanguageStyle
)
from .audio.sources.audiosource import AudioSource

LOG = logging.getLogger(__name__)

# Get the package name dynamically
PACKAGE_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

def load_env_file(env_path=".env"):
    """
    Load environment variables from a .env file.

    Parameters:
    - env_path (str): The path to the .env file. Default is '.env'.

    Returns:
    - bool: True if the file was successfully loaded, False otherwise.
    """
    if not os.path.isfile(env_path):
        print(f"File '{env_path}' does not exist.")
        return False

    try:
        with open(env_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Ensure the line contains a key-value pair
                if "=" not in line:
                    print(f"Ignored invalid line: '{line}'")
                    continue

                # Split the key and value
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Set the environment variable
                os.environ[key] = value

        LOG.info(f"Environment variables from '{env_path}' loaded successfully.")
        return True
    # pylint: disable=broad-exception-caught
    except Exception as e:
        print(f"Error while loading '{env_path}': {e}")
        return False

def timestr_to_sample(timestr: str, sample_rate: int = 16000) -> int:
    """
    Converts a time string in the format hh:mm:ss.ms, mm:ss.ms, or ss.ms
    (milliseconds optional) to the corresponding sample index.

    Args:
        timestr (str): Time string in the format hh:mm:ss.ms, mm:ss.ms, or ss.ms.
        sample_rate (int): Sampling rate in Hz (default is 16000).

    Returns:
        int: The corresponding sample index.
    """
    # Define regex patterns for specific formats
    hh_mm_ss_ms_pattern = re.compile(
        r"^(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$"
    )
    mm_ss_ms_pattern = re.compile(
        r"^(?P<minutes>\d+):(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$"
    )
    ss_ms_pattern = re.compile(
        r"^(?P<seconds>\d+)(?:\.(?P<milliseconds>\d+))?$"
    )

    # Match the input against patterns
    if match := hh_mm_ss_ms_pattern.match(timestr.strip()):
        hours = int(match.group("hours"))
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    elif match := mm_ss_ms_pattern.match(timestr.strip()):
        hours = 0
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    elif match := ss_ms_pattern.match(timestr.strip()):
        hours = 0
        minutes = 0
        seconds = int(match.group("seconds"))
        milliseconds = int(match.group("milliseconds")) if match.group("milliseconds") else 0
    else:
        raise ValueError(f"Invalid time string format: {timestr}")

    # Calculate total time in seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

    # Convert to sample index
    return int(total_seconds * sample_rate)


def configure_writers(config:Config, original_audio_file:str) -> TranscriptWriter:
    # pylint: disable=import-outside-toplevel
    from .transcript.format import (TextTranscriptWriter, MultiTranscriptWriter, AssTranscriptWriter,
        DocxTranscriptWriter, MarkdownTranscriptWriter, JsonTranscriptWriter, StdoutTranscriptWriter)

    multi_writer:MultiTranscriptWriter = MultiTranscriptWriter()
    if config.enable_txt:
        multi_writer.add_writer(TextTranscriptWriter(config=config.write_config))
    if config.enable_ass:
        multi_writer.add_writer(AssTranscriptWriter(config=config.write_config, original_audio_file=original_audio_file))
    if config.enable_docx:
        multi_writer.add_writer(DocxTranscriptWriter(config=config.write_config))
    if config.enable_md:
        multi_writer.add_writer(MarkdownTranscriptWriter(config=config.write_config))
    if config.enable_json:
        multi_writer.add_writer(JsonTranscriptWriter(config=config.write_config))
    if config.enable_stdout and not config.enable_stdout_nocolor:
        multi_writer.add_writer(StdoutTranscriptWriter(config=config.write_config, with_colours=True))
    if config.enable_stdout_nocolor:
        multi_writer.add_writer(StdoutTranscriptWriter(config=config.write_config, with_colours=False))
    return multi_writer

def configure_audio_source(config:Config) -> AudioSource:
    # pylint: disable=import-outside-toplevel
    from .audio.sources.micaudiosource import MicAudioSourcePyAudio as MicAudioSource
    from .audio.sources.fileaudiosource import FileAudioSource
    from .audio.sources.pcmaudiosource import PCMInputStreamAudioSource

    if config.input_source == "-":
        return PCMInputStreamAudioSource(stream=sys.stdin, channels=1, sampling_rate=16000, dtype=np.int16)
    elif config.input_source is None or config.input_source == ">":
        return MicAudioSource()
    else:
        file_audio_source = FileAudioSource(config.input_source, start_sample=config.start_time, end_sample=config.stop_time)
        if not config.stream:
            if config.isolate is not None:
                file_audio_source.isolate_voices(out_path_prefix=config.isolate or None)
            if not config.diarize is None:
                config.diarization = file_audio_source.compute_diarization(
                    rttm_file=config.diarization_file, device=config.device, nb_speakers=config.diarize)

        if config.diarization_file:
            from .voices.diarization import Diarization
            config.diarization = Diarization.load_diarization(rttm_file=config.diarization_file)
        return file_audio_source

def main():
    class OptionalValueAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Set the attribute to the provided value or an empty string if no value is given
            setattr(namespace, self.dest, values if values is not None else "")

    parser = argparse.ArgumentParser(prog=PACKAGE_NAME, description="Verbatim: that's what she said")

    # Specify the command line arguments
    parser.add_argument("input", nargs='?', default=None,
                        help="Path to the input audio file. Use '-' to read from stdin "
                        "(expecting 16bit 16kHz mono PCM stream) or '>' to use microphone.")
    parser.add_argument("-f", "--from",
                        help="Start time within the file in hh:mm:ss.ms or mm:ss.ms", default="00:00.000", dest="start_time")
    parser.add_argument("-t", "--to",
                        help="Stop time within the file in hh:mm:ss.ms or mm:ss.ms", default="", dest="stop_time")
    parser.add_argument("-o", "--outdir",
                        help="Path to the output directory", default=".")
    parser.add_argument("-d", "--diarization", nargs='?', action=OptionalValueAction, default=None,
                        help="Identify speakers in transcript using the diarization RTTM file at the specified path (ex. diarization.rttm)")
    parser.add_argument("-i", "--isolate", nargs='?', action=OptionalValueAction, default=None,
                    help="Extract voices from background noise. Outputs files <name>-vocals.wav and <name>-noise.wav "
                    "if a name is provided, otherwise uses default names.")
    parser.add_argument("-l", "--languages", nargs="*",
                        help="Languages for speech recognition. Provide multiple values for multiple languages.")
    parser.add_argument("-n", "--diarize", nargs='?', action=OptionalValueAction, default=None,
                        help="Number of speakers in the audio file.")
    parser.add_argument("-b", "--nb_beams", type=int,
                        help="Number of parallel when resolving transcription. " +
                        "1-3 for fast, 12-15 for high quality. Default is 9.",
                        default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (specify multiple times for more verbosity)")
    parser.add_argument("--version", action="version", version=f"{PACKAGE_NAME} {__version__}")
    parser.add_argument("--cpu", action="store_true", help="Toggle CPU usage")
    parser.add_argument("-s", "--stream", action="store_true", help="Set mode to low latency streaming")
    parser.add_argument("-w", "--workdir", nargs='?', action=OptionalValueAction, default=None,
                        help="Set the working directory where temporary files may be written to (default is system temp directory)")
    parser.add_argument("--ass", action="store_true", help="Enable ASS subtitle file output")
    parser.add_argument("--docx", action="store_true", help="Enable Microsoft Word DOCX output")
    parser.add_argument("--txt", action="store_true", help="Enable TXT file output")
    parser.add_argument("--json", action="store_true", help="Enable json file output")
    parser.add_argument("--md", action="store_true", help="Enable Markdown (MD) output")
    parser.add_argument("--stdout", action="store_true", default=True, help="Enable stdout output (enabled by default)")
    parser.add_argument("--stdout-nocolor", action="store_true", help="Enable stdout output without colors")
    parser.add_argument("--format-timestamp", type=lambda s: TimestampStyle[s], choices=list(TimestampStyle), default=TimestampStyle.minute,
                        help="Set the timestamp format: 'none' for no timestamps, 'start' for start time, 'range' for start and end times")
    parser.add_argument("--format-speaker", type=lambda s: SpeakerStyle[s], choices=list(SpeakerStyle), default=SpeakerStyle.change,
                        help="Set the timestamp format: 'none' for no timestamps, 'start' for start time, 'range' for start and end times")
    parser.add_argument("--format-probability", type=lambda s: ProbabilityStyle[s], choices=list(ProbabilityStyle), default=ProbabilityStyle.line,
                        help="Set the timestamp format: 'none' for no timestamps, 'start' for start time, 'range' for start and end times")
    parser.add_argument("--format-language", type=lambda s: LanguageStyle[s], choices=list(LanguageStyle), default=LanguageStyle.change,
                        help="Set the timestamp format: 'none' for no timestamps, 'start' for start time, 'range' for start and end times")

    args = parser.parse_args()
    # Set logging level based on verbosity
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(stream=sys.stderr,
                        level=log_level,
                        format='%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%SZ')

    load_env_file()

    # Check if the version option is specified
    if hasattr(args, 'version') and args.version:
        return  # Exit if the version option is specified

    config:Config = Config()
    config.input_source = args.input

    # Set the output directory
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)
    config.output_dir = args.outdir
    LOG.info(f"Output directory set to {config.output_dir}")

    # Set the working directory
    if args.workdir is None:
        config.working_dir = os.getenv("TMPDIR", os.getenv("TEMP", os.getenv("TMP", ".")))
    elif args.workdir == "":
        config.working_dir = config.output_dir
    else:
        if not os.path.isdir(args.workdir):
            os.makedirs(args.workdir)
        config.working_dir = args.workdir
    LOG.info(f"Working directory set to {config.working_dir}")

    input_name_no_ext = os.path.splitext(os.path.split(config.input_source)[-1])[0]
    config.output_prefix_no_ext = os.path.join(config.output_dir, input_name_no_ext)
    config.working_prefix_no_ext = os.path.join(config.working_dir, input_name_no_ext)

    # Set the output directory
    config.start_time = timestr_to_sample(args.start_time) if args.start_time else None
    config.stop_time = timestr_to_sample(args.stop_time) if args.stop_time else None

    # Set output formats
    config.write_config.timestamp_style = args.format_timestamp
    config.write_config.probability_style = args.format_probability
    config.write_config.speaker_style = args.format_speaker
    config.write_config.language_style = args.format_language
    config.write_config.verbose = log_level <= logging.INFO

    config.enable_ass = args.ass
    config.enable_docx = args.docx
    config.enable_txt = args.txt
    config.enable_md = args.md
    config.enable_json = args.json
    config.enable_stdout = args.stdout
    config.enable_stdout_nocolor = args.stdout_nocolor

    if args.cpu or not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Set CUDA_VISIBLE_DEVICES to -1 to use CPU
        LOG.info("Using CPU")
        config.device = "cpu"
    else:
        LOG.info("Using GPU")
        config.device = "cuda"

    config.stream = args.stream
    if config.stream:
        config.chunk_table = [
            (0, 1),
        ]
        config.whisper_best_of = 3
        config.whisper_beam_size = 3
        config.whisper_patience = 3.0
        config.whisper_temperatures = [0, 0.6]

    config.lang = args.languages if args.languages else ["en", "fr"]
    config.isolate = args.isolate
    config.diarize = args.diarize
    if config.diarize == '':
        config.diarize = 0
    elif config.diarize is not None:
        config.diarize = int(config.diarize)

    config.diarization_file = args.diarization
    if config.diarization_file == "" or (config.diarize is not None and config.diarization_file is None):
        config.diarization_file = config.output_prefix_no_ext + ".rttm"

    config.source_stream = configure_audio_source(config)

    writer:TranscriptWriter = configure_writers(config, original_audio_file=config.input_source)

    # pylint: disable=import-outside-toplevel
    from .verbatim import Verbatim
    transcriber = Verbatim(config)
    writer.open(path_no_ext=config.output_prefix_no_ext)
    for utterance, unacknowledged, unconfirmed in transcriber.transcribe():
        writer.write(utterance=utterance, unacknowledged_utterance=unacknowledged, unconfirmed_words=unconfirmed)
    writer.close()

if __name__ == "__main__":
    sys.argv = ['run.py', 'samples/voices.wav', '-w', 'out', '--language', 'en', 'fr', '--txt', '--md', '--json', '--docx', '--ass']
    main()
