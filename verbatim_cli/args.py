"""CLI argument builder shared across verbatim tools."""

import argparse

from verbatim import __version__
from verbatim_files.format.writer import LanguageStyle, ProbabilityStyle, SpeakerStyle, TimestampStyle


class OptionalValueAction(argparse.Action):
    """Action that allows an optional value; if omitted sets an empty string."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values if values is not None else "")


def add_shared_arguments(parser: argparse.ArgumentParser, *, include_input: bool = True, prog: str = "verbatim") -> argparse.ArgumentParser:
    if include_input:
        parser.add_argument(
            "input",
            nargs="?",
            default=None,
            help="Path to the input audio file. Use '-' to read from stdin (expecting 16bit 16kHz mono PCM stream) or '>' to use microphone.",
        )
    parser.add_argument("-f", "--from", default="00:00.000", dest="start_time", help="Start time within the file in hh:mm:ss.ms or mm:ss.ms")
    parser.add_argument("-t", "--to", default="", dest="stop_time", help="Stop time within the file in hh:mm:ss.ms or mm:ss.ms")
    parser.add_argument("-o", "--outdir", default=".", help="Path to the output directory")
    parser.add_argument("--diarize", choices=["pyannote", "energy", "channel", "separate"], default=None, help="Diarization strategy to use")
    parser.add_argument(
        "--diarize-policy",
        help="Channel/range policy like '1,2=energy;3=pyannote;*=channel' (overrides --diarize if set)",
    )
    parser.add_argument(
        "--vttm",
        nargs="?",
        action=OptionalValueAction,
        default=None,
        help="Path to VTTM diarization file; if omitted, a minimal VTTM is created (and filled if diarization runs).",
    )
    parser.add_argument("-d", "--diarization", nargs="?", action=OptionalValueAction, default=None, help="RTTM diarization file path")
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
    parser.add_argument(
        "-n",
        "--speakers",
        nargs="?",
        action=OptionalValueAction,
        default=None,
        help="Number of speakers (int), 'auto' for automatic, or '0' to disable fixed-count hints.",
    )
    parser.add_argument(
        "-b",
        "--nb_beams",
        type=int,
        default=None,
        help="Number of parallel when resolving transcription. 1-3 for fast, 12-15 for high quality. Default is 9.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (specify multiple times for more verbosity)")
    parser.add_argument("--version", action="version", version=f"{prog} {__version__}")
    parser.add_argument("--cpu", action="store_true", help="Toggle CPU usage")
    parser.add_argument("-s", "--stream", action="store_true", help="Set mode to low latency streaming")
    parser.add_argument("--offline", action="store_true", help="Disallow any network/model downloads; use cache only")
    parser.add_argument("--model-cache", default=None, help="Deterministic cache directory for models and downloads")
    parser.add_argument("--config", help="Path to YAML or JSON config file for defaults")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Prefetch commonly used models into the cache and exit",
    )
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
    return parser


def build_parser(prog: str = "verbatim", *, include_input: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Verbatim: that's what she said")
    return add_shared_arguments(parser, include_input=include_input, prog=prog)
