import argparse
import logging
import os
import sys

from .__init__ import __version__
from .config import Config
from .transcript.format.writer import (
    TranscriptWriter,
    SpeakerStyle,
    TimestampStyle,
    ProbabilityStyle,
    LanguageStyle
)

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

    config:Config = Config(
        use_cpu=args.cpu,
        input_source=args.input,
        outdir=args.outdir,
        workdir=args.workdir,
        stream=args.stream,
        isolate=args.isolate,
        diarize=args.diarize,
        start_time=args.start_time,
        stop_time=args.stop_time)

    config.lang = args.languages if args.languages else ["en"]

    # Set output formats
    config.write_config.timestamp_style = args.format_timestamp
    config.write_config.probability_style = args.format_probability
    config.write_config.speaker_style = args.format_speaker
    config.write_config.language_style = args.format_language
    config.write_config.verbose = log_level <= logging.INFO

    output_formats = []
    if args.ass:
        output_formats.append("ass")
    if args.docx:
        output_formats.append("docx")
    if args.txt:
        output_formats.append("txt")
    if args.md:
        output_formats.append("md")
    if args.json:
        output_formats.append("json")
    if args.stdout:
        output_formats.append("stdout")
    if args.stdout_nocolor:
        output_formats.append("stdout-nocolor")

    config.configure_output_formats(output_formats=output_formats)

    # pylint: disable=import-outside-toplevel
    from .verbatim import Verbatim
    transcriber = Verbatim(config)
    writer:TranscriptWriter = configure_writers(config, original_audio_file=config.input_source)
    writer.open(path_no_ext=config.output_prefix_no_ext)
    for utterance, unacknowledged, unconfirmed in transcriber.transcribe():
        writer.write(utterance=utterance, unacknowledged_utterance=unacknowledged, unconfirmed_words=unconfirmed)
    writer.close()

if __name__ == "__main__":
    sys.argv = ['run.py', 'samples/voices.wav', '-w', 'out', '--language', 'en', 'fr', '--txt', '--md', '--json', '--docx', '--ass']
    main()
