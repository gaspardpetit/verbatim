from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from systems import BENCHMARK_CONFIG_PATH, SYSTEMS_CONFIG_PATH, load_benchmark_plan  # noqa: E402  # pylint: disable=wrong-import-position


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SwitchLingua benchmark sequentially across languages and systems.")
    parser.add_argument("--python", default=None, help="Python executable to use. Defaults to the current interpreter.")
    parser.add_argument("--benchmark-config", default=str(BENCHMARK_CONFIG_PATH), help="YAML file defining default languages and systems")
    parser.add_argument("--systems-config", default=str(SYSTEMS_CONFIG_PATH), help="YAML file defining benchmark systems")
    parser.add_argument("--languages", nargs="*", default=None, help="Languages to run in sequence.")
    parser.add_argument("--systems", nargs="*", default=None, help="Systems to run for each language.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with remaining runs after a failure.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Forward -v / -vv to benchmark.py")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra arguments forwarded to benchmark.py")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    python_exe = args.python or sys.executable
    benchmark_script = Path(__file__).with_name("benchmark.py")
    plan = load_benchmark_plan(Path(args.benchmark_config), systems_config_path=Path(args.systems_config))
    languages = list(args.languages) if args.languages else plan["languages"]
    systems = list(args.systems) if args.systems else plan["systems"]

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    failures: list[tuple[str, str, int]] = []
    for language in languages:
        for system in systems:
            command = [
                python_exe,
                str(benchmark_script),
                "--systems-config",
                args.systems_config,
                "--lang",
                language,
                "--systems",
                system,
                "--skip-existing",
            ]
            if args.continue_on_error:
                command.append("--continue-on-error")
            if args.verbose:
                command.append("-" + ("v" * args.verbose))
            command.extend(extra_args)

            print(f"=== {language} / {system} ===")
            print(" ".join(command))
            completed = subprocess.run(command, check=False)  # nosec B603
            if completed.returncode != 0:
                failures.append((language, system, completed.returncode))
                if not args.continue_on_error:
                    return completed.returncode

    if failures:
        print("\nFailures:")
        for language, system, returncode in failures:
            print(f"- {language} / {system}: exit {returncode}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
