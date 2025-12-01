import argparse
import logging
import sys

from verbatim_serve.server import serve

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(prog="verbatim-serve", description="Verbatim HTTP server")
    parser.add_argument("--cpu", action="store_true", help="Force CPU device")
    parser.add_argument("--model-cache", default=None, help="Deterministic cache directory for models and downloads")
    parser.add_argument("--offline", action="store_true", help="Disallow any network/model downloads; use cache only")
    parser.add_argument(
        "-w",
        "--workdir",
        nargs="?",
        default=None,
        help="Working directory where temporary files may be written (default: system temp directory)",
    )
    parser.add_argument("-o", "--outdir", default=".", help="Path to the output directory")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (specify multiple times for more verbosity)")

    args = parser.parse_args()
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

    from verbatim.config import Config

    config: Config = Config(
        device="cpu" if args.cpu else "auto",
        output_dir=args.outdir,
        working_dir=args.workdir if args.workdir is not None else "",
        stream=False,
        offline=args.offline,
        model_cache_dir=args.model_cache,
    )

    serve(config)


if __name__ == "__main__":
    main()
