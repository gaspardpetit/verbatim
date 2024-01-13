import argparse
import os
import logging
from .__init__ import __version__
from .pipeline import Pipeline
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
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (specify multiple times for more verbosity)")
    parser.add_argument("--version", action="version", version=f"{package_name} {__version__}")

    args = parser.parse_args()

    # Set logging level based on verbosity
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = log_levels[min(args.verbose, len(log_levels) - 1)]
    logging.basicConfig(level=log_level,
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

    # Validate output directory existence
    if not os.path.isdir(args.output):
        print(f"Error: Output directory '{args.output}' not found.")
        return

    # Use the provided values or default values for languages and nb_speakers
    languages = args.languages if args.languages else ["en", "fr"]
    nb_speakers = args.nb_speakers

    context: Context = Context(
        languages=languages, nb_speakers=nb_speakers, source_file=args.input, out_dir=args.output, log_level=log_level)
    pipeline: Pipeline = Pipeline(context=context)
    pipeline.execute()

if __name__ == "__main__":
    main()
