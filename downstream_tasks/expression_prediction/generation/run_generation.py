#!/usr/bin/env python3
"""Run one config-driven gena_expression sequence-generation experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.config import RunConfig
from utils.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    """Parse the path to one experiment configuration."""

    parser = argparse.ArgumentParser(
        description="Sample genomic promoters, evolve them in plasmid context, and save results."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="YAML configuration file. Relative data paths are resolved from this file.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the requested configuration and execute one experiment."""

    args = parse_args()
    config = RunConfig.load(args.config)
    written = run_experiment(config)
    print("Written files:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
