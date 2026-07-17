#!/usr/bin/env python3
"""Discover the last checkpoint of every finetuning run under a runs/ tree.

Each run directory (the one holding config.json, written by train.py) is
scanned for its highest-step `checkpoint-<N>` subdirectory that actually
contains `model.safetensors`. The pretrained backbone is read straight out of
that config.json's `from_pretrained` field instead of being guessed from the
directory name.

max_length is fixed at DEFAULT_MAX_LENGTH -- this server only evaluates
ModernGENA checkpoints, whose DNA-trained BPE covers a full 3072bp training
chunk well within that budget (see
analysis/tokenizer_sequence_length_analysis.ipynb). If a ModernBERT-backbone
run ever needs evaluating here, pass a larger --max-length to
evaluate_all_mammals.sh for that checkpoint by hand.

Output (stdout): one TSV row per run -- last-checkpoint manifest, ready
to be consumed by slurm/mammals_eval.slurm as one array-job task per line:
    model_path<TAB>pretrained_config_name<TAB>max_length<TAB>experiment_name
"""

import argparse
import fnmatch
import json
import re
import sys
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

DEFAULT_MAX_LENGTH = 512


def find_runs(runs_root: Path, pattern: str):
    for config_path in sorted(runs_root.rglob("config.json")):
        run_dir = config_path.parent
        rel = run_dir.relative_to(runs_root)
        top = rel.parts[0]
        if pattern and not fnmatch.fnmatch(top, pattern):
            continue
        yield rel, run_dir, config_path


def last_checkpoint(run_dir: Path):
    best_step, best_dir = -1, None
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = CHECKPOINT_RE.match(child.name)
        if not m:
            continue
        if not (child / "model.safetensors").exists():
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step, best_dir = step, child
    return best_step, best_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", nargs="?", default="runs", help="Root dir to scan (default: runs)")
    parser.add_argument(
        "--pattern",
        default="mammals_contig_separated*",
        help="fnmatch pattern applied to the top-level dir name under runs_root "
        "(default: 'mammals_contig_separated*', i.e. the multi-species mammals runs "
        "that evaluate_all_mammals.sh's species lists are meant for -- this excludes "
        "the single-species HUMAN_CONTIGS/MOUSE/human_mouse_contigs/mouse_human_contigs runs). "
        "Pass '*' to include every run under runs_root.",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        print(f"error: runs root '{runs_root}' is not a directory", file=sys.stderr)
        sys.exit(1)

    n_found, n_skipped = 0, 0
    for rel, run_dir, config_path in find_runs(runs_root, args.pattern):
        try:
            cli_args = json.loads(config_path.read_text())["cli_args"]
            from_pretrained = cli_args["from_pretrained"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"skip {rel}: unreadable config.json ({e})", file=sys.stderr)
            n_skipped += 1
            continue

        step, ckpt_dir = last_checkpoint(run_dir)
        if ckpt_dir is None:
            print(f"skip {rel}: no checkpoint-<N>/model.safetensors found", file=sys.stderr)
            n_skipped += 1
            continue

        max_length = DEFAULT_MAX_LENGTH
        model_path = ckpt_dir / "model.safetensors"
        experiment_name = "__".join(rel.parts) + f"__checkpoint-{step}"

        print(f"{model_path}\t{from_pretrained}\t{max_length}\t{experiment_name}")
        print(f"found {rel} -> checkpoint-{step} ({from_pretrained}, max_length={max_length})", file=sys.stderr)
        n_found += 1

    print(f"\n{n_found} checkpoint(s) found, {n_skipped} run(s) skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
