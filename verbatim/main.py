import argparse
import logging
import os
import sys
from typing import List

from .__init__ import __version__
from .transcript.format.writer import (
    LanguageStyle,
    ProbabilityStyle,
    SpeakerStyle,
    TimestampStyle,
)

LOG = logging.getLogger(__name__)

# Get the package name dynamically
PACKAGE_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


def main():
    # pylint: disable=import-outside-toplevel
    class OptionalValueAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Set the attribute to the provided value or an empty string if no value is given
            setattr(namespace, self.dest, values if values is not None else "")

    parser = argparse.ArgumentParser(prog=PACKAGE_NAME, description="Verbatim: that's what she said")

    # Specify the command line arguments
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the input audio file. Use '-' to read from stdin (expecting 16bit 16kHz mono PCM stream) or '>' to use microphone.",
    )
    parser.add_argument("-f", "--from", default="00:00.000", dest="start_time", help="Start time within the file in hh:mm:ss.ms or mm:ss.ms")
    parser.add_argument("-t", "--to", default="", dest="stop_time", help="Stop time within the file in hh:mm:ss.ms or mm:ss.ms")
    parser.add_argument("-o", "--outdir", default=".", help="Path to the output directory")
    parser.add_argument("--diarization-strategy", choices=["pyannote", "stereo"], default="pyannote", help="Diarization strategy to use")
    parser.add_argument(
        "--vttm",
        nargs="?",
        action=OptionalValueAction,
        default=None,
        help="Path to VTTM diarization file; if omitted, a minimal VTTM is created (and filled if diarization runs).",
    )
    parser.add_argument(
        "-d",
        "--diarization",
        nargs="?",
        action=OptionalValueAction,
        default=None,
        help="(Deprecated) RTTM diarization file path; will be wrapped into a VTTM for processing.",
    )
    parser.add_argument(
        "--separate", nargs="?", action=OptionalValueAction, default=None, help="Enables speaker voice separation and process each speaker separately"
    )
    parser.add_argument(
        "-i",
        "--isolate",
        nargs="?",
        action=OptionalValueAction,
        default=None,
        help="Extract voices from background noise. Outputs files <name>-vocals.wav and <name>-noise.wav "
        "if a name is provided, otherwise uses default names.",
    )
    parser.add_argument("-l", "--languages", nargs="*", help="Languages for speech recognition. Provide multiple values for multiple languages.")
    parser.add_argument("-n", "--diarize", nargs="?", action=OptionalValueAction, default=None, help="Number of speakers in the audio file.")
    parser.add_argument(
        "-b",
        "--nb_beams",
        type=int,
        default=None,
        help="Number of parallel when resolving transcription. 1-3 for fast, 12-15 for high quality. Default is 9.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (specify multiple times for more verbosity)")
    parser.add_argument("--version", action="version", version=f"{PACKAGE_NAME} {__version__}")
    parser.add_argument("--cpu", action="store_true", help="Toggle CPU usage")
    parser.add_argument("-s", "--stream", action="store_true", help="Set mode to low latency streaming")
    parser.add_argument("--offline", action="store_true", help="Disallow any network/model downloads; use cache only")
    parser.add_argument("--model-cache", default=None, help="Deterministic cache directory for models and downloads")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Prefetch commonly used models into the cache and exit",
    )
    parser.add_argument("--serve", action="store_true", help="Start an HTTP server with an OpenAI-compatible endpoint")
    parser.add_argument(
        "-w",
        "--workdir",
        nargs="?",
        action=OptionalValueAction,
        default=None,
        help="Set the working directory where temporary files may be written to (default is system temp directory)",
    )
    parser.add_argument("--ass", action="store_true", help="Enable ASS subtitle file output")
    parser.add_argument("--docx", action="store_true", help="Enable Microsoft Word DOCX output")
    parser.add_argument("--txt", action="store_true", help="Enable TXT file output")
    parser.add_argument("--json", action="store_true", help="Enable json file output")
    parser.add_argument("--md", action="store_true", help="Enable Markdown (MD) output")
    parser.add_argument("--stdout", action="store_true", default=True, help="Enable stdout output (enabled by default)")
    parser.add_argument("--stdout-nocolor", action="store_true", help="Enable stdout output without colors")
    parser.add_argument(
        "--format-timestamp",
        type=lambda s: TimestampStyle[s],
        choices=list(TimestampStyle),
        default=TimestampStyle.minute,
        help="Set the timestamp format: 'none' for no timestamps, 'start' for start time, 'range' for start and end times",
    )
    parser.add_argument(
        "--format-speaker",
        type=lambda s: SpeakerStyle[s],
        choices=list(SpeakerStyle),
        default=SpeakerStyle.change,
        help=(
            "Set the speaker format: 'none' for no speaker tags, "
            "'change' to show the speaker only when it changes, "
            "'always' to prefix every line with the speaker"
        ),
    )
    parser.add_argument(
        "--format-probability",
        type=lambda s: ProbabilityStyle[s],
        choices=list(ProbabilityStyle),
        default=ProbabilityStyle.line,
        help=(
            "Set the probability format: choose between 'line' or 'word' "
            "styles with optional thresholds (none, line, line_75, line_50, "
            "line_25, word, word_75, word_50, word_25)"
        ),
    )
    parser.add_argument(
        "--format-language",
        type=lambda s: LanguageStyle[s],
        choices=list(LanguageStyle),
        default=LanguageStyle.change,
        help=(
            "Set the language format: 'none' for no language tags, "
            "'change' to display language when it changes, "
            "'always' to show language on each line"
        ),
    )
    parser.add_argument("-e", "--eval", nargs="?", default=None, help="Path to reference json file")

    args = parser.parse_args()
    # Set logging level based on verbosity
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format="%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # load the values from the .env file, if present
    from .env import load_env_file

    load_env_file()

    # Check if the version option is specified
    if hasattr(args, "version") and args.version:
        return  # Exit if the version option is specified

    from .config import Config

    config: Config = Config(
        device="cpu" if args.cpu else "auto",
        output_dir=args.outdir,
        working_dir=args.workdir if args.workdir is not None else "",
        stream=args.stream,
        offline=args.offline,
        model_cache_dir=args.model_cache,
    )

    config.lang = args.languages if args.languages else ["en"]

    # Handle install-only mode
    if args.install:
        LOG.info("Installing/prefetching models into cache...")
        from .prefetch import prefetch

        prefetch(model_cache_dir=args.model_cache, whisper_size=config.whisper_model_size)
        LOG.info("Model prefetch complete.")
        return

    if args.serve:
        from .server import serve

        serve(config)
        return

    source_path = args.input
    input_name_no_ext = os.path.splitext(os.path.split(source_path)[-1])[0]
    output_prefix_no_ext = os.path.join(config.output_dir, input_name_no_ext)
    working_prefix_no_ext = os.path.join(config.working_dir, input_name_no_ext)

    # Set output formats
    from .transcript.format.writer import TranscriptWriterConfig

    write_config: TranscriptWriterConfig = TranscriptWriterConfig()
    write_config.timestamp_style = args.format_timestamp
    write_config.probability_style = args.format_probability
    write_config.speaker_style = args.format_speaker
    write_config.language_style = args.format_language
    write_config.verbose = log_level <= logging.INFO

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

    if args.diarize == "":
        diarize = 0
    elif args.diarize is None:
        diarize = None
    else:
        diarize = int(args.diarize)

    from .audio.sources.sourceconfig import SourceConfig

    source_config: SourceConfig = SourceConfig(
        isolate=args.isolate,
        diarize=diarize,
        diarization_file=args.diarization,
        diarization_strategy=args.diarization_strategy,
    )

    from .audio.sources.audiosource import AudioSource

    audio_sources: List[AudioSource] = []

    if args.separate:
        # perform the transcription by combining the transcript of
        # multiple audio sources separated from a single one
        from .audio.sources.factory import create_separate_speaker_sources

        audio_sources += create_separate_speaker_sources(
            strategy=args.separate or "pyannote",
            source_config=source_config,
            device=config.device,
            input_source=source_path,
            start_time=args.start_time,
            stop_time=args.stop_time,
            working_prefix_no_ext=working_prefix_no_ext,
            output_prefix_no_ext=output_prefix_no_ext,
        )
    else:
        from .audio.sources.factory import create_audio_source

        audio_sources.append(
            create_audio_source(
                source_config=source_config,
                device=config.device,
                input_source=source_path,
                start_time=args.start_time,
                stop_time=args.stop_time,
                working_prefix_no_ext=working_prefix_no_ext,
                output_prefix_no_ext=output_prefix_no_ext,
                stream=config.stream,
            )
        )

    from verbatim.verbatim import execute

    execute(
        source_path=source_path,
        config=config,
        write_config=write_config,
        audio_sources=audio_sources,
        output_formats=output_formats,
        output_prefix_no_ext=output_prefix_no_ext,
        working_prefix_no_ext=working_prefix_no_ext,
        eval_file=args.eval,
    )


if __name__ == "__main__":
    main()
