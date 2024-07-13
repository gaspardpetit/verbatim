import argparse
import os
import sys
import logging
import torch
from .__init__ import __version__
from .context import Context


LOG = logging.getLogger(__name__)

# Get the package name dynamically
package_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(prog=package_name, description="Verbatim: that's what she said")

    # Specify the command line arguments
    parser.add_argument("input",
                        help="Path to the input audio file")
    parser.add_argument("-o", "--output",
                        help="Path to the output directory", default=".")
    parser.add_argument("-l", "--languages", nargs="*",
                        help="Languages for speech recognition. Provide multiple values for multiple languages.")
    parser.add_argument("-n", "--nb_speakers", type=int,
                        help="Number of speakers in the audio file. Defaults to 1.", default=1)
    parser.add_argument("-b", "--nb_beams", type=int,
                        help="Number of parallel when resolving transcription. " +
                        "1-3 for fast, 12-15 for high quality. Default is 9.",
                        default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (specify multiple times for more verbosity)")
    parser.add_argument("--version", action="version", version=f"{package_name} {__version__}")
    parser.add_argument("--cpu", action="store_true", help="Toggle CPU usage")
    parser.add_argument("--transcribe_only", action="store_true",
                        help="Skip preprocessing, including diarization and only transcribe")
    parser.add_argument("--write-config",
                        help="Write the config file so it can be edited", default=None)
    parser.add_argument("--read-config",
                        help="Read a config file", default=None)

    args = parser.parse_args()

    # Set logging level based on verbosity
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
    logging.basicConfig(stream=sys.stderr,
                        level=log_level,
                        format='%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d][%(funcName)s] %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%SZ')

    # Check if the version option is specified
    if hasattr(args, 'version') and args.version:
        return  # Exit if the version option is specified

    # Print help if no input is provided
    if not args.input:
        parser.print_help()
        return

    # Validate input file existence
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    # Validate output directory existence or create it
    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    # Create output directory if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.cpu or not torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Set CUDA_VISIBLE_DEVICES to -1 to use CPU
        print("Using CPU")
        device = "cpu"
    else:
        print("Using GPU")
        device = "cuda"

    # Use the provided values or default values for languages and nb_speakers
    languages = args.languages if args.languages else ["en", "fr"]
    nb_speakers = args.nb_speakers

    if args.read_config:
        with open(args.read_config, "r", encoding="utf-8") as f:
            text = f.read()
            context: Context = Context.from_yaml(source_file=args.input, out_dir=args.output, text=text)
    else:
        context: Context = Context(source_file=args.input, out_dir=args.output)

    context.languages=languages
    context.nb_speakers=nb_speakers
    context.log_level=log_level
    context.device=device

    if args.nb_beams is None:
        if context.beams is None:
            context.beams = 9
    else:
        context.beams = args.nb_beams

    if args.transcribe_only:
        context.transcribe_only = True
    else:
        context.transcribe_only = False


    if args.write_config:
        with open(args.write_config, "w", encoding="utf-8") as f:
            f.write(context.to_yaml())

    if context.transcribe_only:
        from .engine import Engine
        from .processor import Processor, ProcessorTranscribe
        engine:Engine = Engine()
        processor:Processor = ProcessorTranscribe(context=context, engine=engine)
        processor.execute()
    else:
        from .pipeline import Pipeline
        pipeline: Pipeline = Pipeline(context=context)
        pipeline.execute()

if __name__ == "__main__":
    main()
