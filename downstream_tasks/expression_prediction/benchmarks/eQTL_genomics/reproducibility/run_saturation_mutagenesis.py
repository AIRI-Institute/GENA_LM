#!/usr/bin/env python3
"""CLI for genome-wide TSS saturation mutagenesis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE.parent
TASK_DIR = HERE.parents[2]
REPO_ROOT = HERE.parents[4]
API_SRC = TASK_DIR / "api" / "src"
for path in (str(REPO_ROOT), str(TASK_DIR), str(API_SRC), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from saturation_mutagenesis import merge_shards, run_worker


def requested_inference_config() -> Path:
    """Find --inference-config before constructing command-specific parsers."""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--inference-config", type=Path)
    known, _ = pre_parser.parse_known_args()
    return known.inference_config or BENCHMARK_DIR / "inference_config.yaml"


def inference_defaults(path: Path) -> dict[str, object]:
    """Read inference-specific defaults that may be overridden on the CLI."""

    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    values = {
        "checkpoint": OmegaConf.select(config, "checkpoint"),
        "genome_fasta": OmegaConf.select(config, "genome_fasta"),
        "description_json": OmegaConf.select(config, "description_json"),
        "mutation_window_bp": OmegaConf.select(config, "mutation_window_bp"),
        "score_window_bp": OmegaConf.select(config, "score_window_bp"),
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    values["dna_input_seq_len"] = OmegaConf.select(
        config, "dna_input_seq_len", default=1022
    )
    values["num_before"] = OmegaConf.select(config, "num_before", default=510)
    return values


def config_path(value: object) -> Path:
    """Resolve inference-config paths relative to the reproducibility directory."""

    path = Path(str(value))
    return path if path.is_absolute() else HERE / path


def shared(parser: argparse.ArgumentParser) -> None:
    inference_config = requested_inference_config()
    defaults = inference_defaults(inference_config)
    parser.add_argument("--inference-config", type=Path, default=inference_config)
    parser.add_argument(
        "--catalog", type=Path, default=BENCHMARK_DIR / "data" / "selected_tss_catalog.tsv"
    )
    parser.add_argument("--genome-fasta", type=Path, default=config_path(defaults["genome_fasta"]))
    parser.add_argument(
        "--description-json", type=Path, default=config_path(defaults["description_json"])
    )
    parser.add_argument("--checkpoint", type=Path, default=config_path(defaults["checkpoint"]))
    parser.add_argument(
        "--mutation-window-bp", type=int, default=int(defaults["mutation_window_bp"])
    )
    parser.add_argument("--score-window-bp", type=int, default=int(defaults["score_window_bp"]))
    parser.add_argument("--score-center", choices=("tss", "variant"), default="tss")
    parser.add_argument("--variant-score-width-bp", type=int, default=501)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--preprocessing-workers", type=int, default=10)
    parser.add_argument("--preprocessing-backend", choices=("process", "thread"), default="process")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--fetch-bp-per-token", type=int, default=20)
    parser.add_argument(
        "--dna-input-seq-len", type=int, default=int(defaults["dna_input_seq_len"])
    )
    parser.add_argument("--num-before", type=int, default=int(defaults["num_before"]))
    parser.add_argument("--on-error", choices=("raise", "record"), default="raise")
    parser.add_argument("--limit", type=int)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot", help="Run a deterministic small benchmark")
    shared(pilot)
    pilot.add_argument("--device", default="cuda:0")
    pilot.set_defaults(limit=2, shard_index=0, shard_count=1)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    shared(worker)
    worker.add_argument("--device", required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--shard-count", type=int, required=True)
    run = commands.add_parser("run", help="Launch one deterministic worker per GPU")
    shared(run)
    run.add_argument("--devices", default="0", help="Comma-separated CUDA device indices")
    merge = commands.add_parser("merge", help="Merge complete compatible shards")
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return root


def launch(args: argparse.Namespace) -> None:
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one GPU index")
    processes = []
    for index, device in enumerate(devices):
        command = [sys.executable, str(Path(__file__).resolve()), "worker"]
        for key, value in vars(args).items():
            if key in {"command", "devices"} or value is None:
                continue
            option = "--" + key.replace("_", "-")
            command.extend((option, str(value)))
        command.extend(("--device", f"cuda:{device}", "--shard-index", str(index), "--shard-count", str(len(devices))))
        processes.append(subprocess.Popen(command))
    failures = [process.wait() for process in processes]
    if any(failures):
        raise SystemExit(f"Worker exit codes: {failures}")


def main() -> None:
    args = parser().parse_args()
    if args.command in {"pilot", "worker"}:
        run_worker(args)
    elif args.command == "run":
        launch(args)
    else:
        merge_shards(sorted(args.output_dir.glob("shard-*.h5")), args.output)


if __name__ == "__main__":
    main()
