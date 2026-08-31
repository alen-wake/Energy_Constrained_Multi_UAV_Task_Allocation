"""Command-line entry point for the four formal paired experiments."""

import argparse
from datetime import datetime
from pathlib import Path

from batch_experiments import run_batch_experiments
from experiment_config import DEFAULT_OUTPUT_DIRECTORY


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run four methods across four paired experiment families"
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=None,
        help="Output directory; a timestamped directory is used by default",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    output_directory = arguments.output or (
        DEFAULT_OUTPUT_DIRECTORY
        / datetime.now().strftime("experiment_results_%Y%m%d_%H%M%S")
    )
    run_batch_experiments(output_directory.resolve())
    print(f"Completed. Output directory: {output_directory.resolve()}")


if __name__ == "__main__":
    main()
