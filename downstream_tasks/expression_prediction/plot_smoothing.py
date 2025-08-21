#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add SMA-smoothed copies of selected TensorBoard scalar tags for every event file under a log directory.

Usage:
  python plot_smoothing.py --logdir /path/to/runs --window 100
  # or with explicit tag list:
  python plot_smoothing.py --logdir /path/to/runs --window 50 \
    --tags \
    "score_predictions_Expression_dataset_v1_mm10_CPM dataset/iterations/valid" \
    "score_predictions_Expression_dataset_v1_mm10_CPM dataset/samples/valid" \
    "score_predictions_Expression_dataset_v1_GRCh38_csv dataset/iterations/valid" \
    "score_predictions_Expression_dataset_v1_GRCh38_csv dataset/samples/valid"

Notes:
- Writing via SummaryWriter will always create a NEW `events.out.tfevents.*` file in the same run directory.
- This script does not modify existing event files.
"""

# pip install tensorboard pandas torch
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import argparse
import pathlib
import os
import sys

DEFAULT_TAGS = [
    # mouse
    "score_predictions_Expression_dataset_v1_mm10_CPM dataset/iterations/valid",
    "score_predictions_Expression_dataset_v1_mm10_CPM dataset/samples/valid",
    # human
    "score_predictions_Expression_dataset_v1_GRCh38_csv dataset/iterations/valid",
    "score_predictions_Expression_dataset_v1_GRCh38_csv dataset/samples/valid",
]

def iter_event_files(logdir: str):
    """Yield absolute paths to TensorBoard event files under `logdir` recursively."""
    for root, _, files in os.walk(logdir):
        for f in files:
            if f.startswith("events.out.tfevents."):
                yield os.path.join(root, f)

def add_sma_scalars_to_tb(
    event_file: str,
    tags,
    window: int = 10,
    suffix_tpl: str = " [slow-{window}]",
) -> int:
    """
    Read scalars from `event_file`, compute SMA with `window`, and write smoothed
    series with a suffix into the SAME run directory (creates a new events file).
    Returns the number of written datapoints.
    """
    if isinstance(tags, str):
        tags = [tags]

    if not os.path.isfile(event_file):
        print(f"[WARN] not a file: {event_file}", file=sys.stderr)
        return 0

    run_dir = os.path.dirname(event_file)

    # Load events from the file
    try:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
    except Exception as e:
        print(f"[WARN] failed to read {event_file}: {e}", file=sys.stderr)
        return 0

    available = set(ea.Tags().get("scalars", []))
    wanted = [t for t in tags if t in available]
    if not wanted:
        # No matching tags in this event file
        return 0

    writer = SummaryWriter(log_dir=run_dir)
    n_points = 0

    for tag in wanted:
        events = ea.Scalars(tag)  # list of ScalarEvent(wall_time, step, value)
        if not events:
            continue

        steps = [e.step for e in events]
        values = [e.value for e in events]
        wall_times = [e.wall_time for e in events]

        # SMA with min_periods=1 to start from the first point
        sma = pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()

        new_tag = f"{tag}{suffix_tpl.format(window=window)}"

        # Optional: skip if this exact smoothed tag already exists in THIS event file
        if new_tag in available:
            print(f"[INFO] tag already present in this file, skip: {new_tag}")
            continue

        for s, v, wt in zip(steps, sma, wall_times):
            writer.add_scalar(new_tag, float(v), global_step=s, walltime=wt)
            n_points += 1

        print(f"[OK] {event_file}: wrote {len(sma)} pts -> '{new_tag}'")

    writer.flush()
    writer.close()
    return n_points

def main():
    parser = argparse.ArgumentParser(
        description="Add SMA-smoothed copies of selected TensorBoard scalar tags into all event files under a logdir."
    )
    parser.add_argument("--logdir", required=True, help="TensorBoard log directory (recursively scanned)")
    parser.add_argument("--window", type=int, default=10, help="SMA window size")
    parser.add_argument(
        "--tags",
        nargs="*",
        default=DEFAULT_TAGS,
        help="List of scalar tags to smooth (default: 4 Expression tags).",
    )
    parser.add_argument(
        "--suffix",
        default=" [slow-{window}]",
        help="Suffix template for new tags (format with {window}).",
    )
    args = parser.parse_args()

    logdir = os.path.abspath(args.logdir)
    if not os.path.isdir(logdir):
        print(f"[ERR] not a directory: {logdir}", file=sys.stderr)
        sys.exit(2)

    total_files = 0
    total_points = 0
    for ev in iter_event_files(logdir):
        total_files += 1
        total_points += add_sma_scalars_to_tb(
            ev, tags=args.tags, window=args.window, suffix_tpl=args.suffix
        )

    print(f"\nDone. Scanned files: {total_files}, written points: {total_points}")
    if total_points == 0:
        print("No matching tags found or nothing to write.")

if __name__ == "__main__":
    main()
