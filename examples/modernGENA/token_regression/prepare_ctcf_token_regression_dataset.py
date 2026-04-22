import argparse
import bisect
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pysam


Interval = Tuple[int, int]
PeakRecord = Tuple[str, int, int, int]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare interval CSVs for token-level CTCF regression."
    )
    parser.add_argument("--fasta_path", type=str, required=True)
    parser.add_argument("--peaks_bed", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--interval_length", type=int, default=10000)
    parser.add_argument("--shift_bp", type=int, default=5000)
    parser.add_argument("--test_chromosomes", type=str, default="chr21")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Cap positives per split (0 means all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def load_peaks(path: Path) -> List[PeakRecord]:
    peaks: List[PeakRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            chrom = normalize_chrom(fields[0])
            start = int(fields[1])
            end = int(fields[2])
            summit_offset = -1
            if len(fields) >= 10:
                try:
                    summit_offset = int(fields[9])
                except ValueError:
                    summit_offset = -1
            center = start + summit_offset if summit_offset >= 0 else (start + end) // 2
            peaks.append((chrom, start, end, center))
    return peaks


def overlaps(intervals: Sequence[Interval], start: int, end: int) -> bool:
    if not intervals:
        return False
    starts = [x[0] for x in intervals]
    idx = bisect.bisect_left(starts, start)
    for i in (idx - 1, idx):
        if 0 <= i < len(intervals):
            s, e = intervals[i]
            if start < e and end > s:
                return True
    return False


def build_peak_intervals(peaks: Sequence[PeakRecord]) -> Dict[str, List[Interval]]:
    by_chrom: Dict[str, List[Interval]] = defaultdict(list)
    for chrom, start, end, _ in peaks:
        by_chrom[chrom].append((start, end))
    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda x: x[0])
    return by_chrom


def build_positive_intervals(
    fasta: pysam.FastaFile,
    peaks: Sequence[PeakRecord],
    all_peak_intervals: Dict[str, List[Interval]],
    interval_length: int,
    shift_bp: int,
    rng: random.Random,
) -> List[dict]:
    result: List[dict] = []
    half = interval_length // 2
    for chrom, _, _, center in peaks:
        if chrom not in fasta.references:
            continue
        chrom_len = fasta.get_reference_length(chrom)
        shift = rng.randint(-shift_bp, shift_bp)
        shifted_center = center + shift
        start = max(0, shifted_center - half)
        end = start + interval_length
        if end > chrom_len:
            end = chrom_len
            start = max(0, end - interval_length)
        if end - start < interval_length:
            continue
        if not overlaps(all_peak_intervals.get(chrom, []), start, end):
            continue
        # Optional sanity fetch through pysam to verify boundaries and bases.
        seq = fasta.fetch(chrom, start, end).upper()
        if len(seq) != interval_length:
            continue
        result.append({"chrom": chrom, "start": start, "end": end, "label": 1})
    return result


def sample_negative_intervals(
    fasta: pysam.FastaFile,
    positives: Sequence[dict],
    all_peak_intervals: Dict[str, List[Interval]],
    interval_length: int,
    rng: random.Random,
) -> List[dict]:
    negatives: List[dict] = []
    counts_by_chrom: Dict[str, int] = defaultdict(int)
    for row in positives:
        counts_by_chrom[row["chrom"]] += 1

    for chrom, target_count in counts_by_chrom.items():
        chrom_len = fasta.get_reference_length(chrom)
        if chrom_len <= interval_length:
            continue
        sampled = 0
        tries = 0
        max_tries = max(2000, target_count * 500)
        exclusion = all_peak_intervals.get(chrom, [])
        while sampled < target_count and tries < max_tries:
            tries += 1
            start = rng.randint(0, chrom_len - interval_length)
            end = start + interval_length
            if overlaps(exclusion, start, end):
                continue
            negatives.append({"chrom": chrom, "start": start, "end": end, "label": 0})
            sampled += 1
    return negatives


def write_csv(path: Path, rows: Sequence[dict]):
    fields = ["chrom", "start", "end", "label", "split"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    peaks = load_peaks(Path(args.peaks_bed).expanduser().absolute())
    if not peaks:
        raise RuntimeError("No peaks loaded from BED.")
    all_peak_intervals = build_peak_intervals(peaks)

    test_chroms = {
        normalize_chrom(ch.strip()) for ch in args.test_chromosomes.split(",") if ch.strip()
    }
    train_peaks = [p for p in peaks if p[0] not in test_chroms]
    test_peaks = [p for p in peaks if p[0] in test_chroms]
    if not test_peaks:
        raise RuntimeError("No peaks found in test chromosomes.")

    with pysam.FastaFile(str(Path(args.fasta_path).expanduser().absolute())) as fasta:
        train_pos = build_positive_intervals(
            fasta=fasta,
            peaks=train_peaks,
            all_peak_intervals=all_peak_intervals,
            interval_length=args.interval_length,
            shift_bp=args.shift_bp,
            rng=rng,
        )
        test_pos = build_positive_intervals(
            fasta=fasta,
            peaks=test_peaks,
            all_peak_intervals=all_peak_intervals,
            interval_length=args.interval_length,
            shift_bp=args.shift_bp,
            rng=rng,
        )

        if args.max_samples and args.max_samples > 0:
            rng.shuffle(train_pos)
            rng.shuffle(test_pos)
            train_pos = train_pos[: args.max_samples]
            test_pos = test_pos[: args.max_samples]

        train_neg = sample_negative_intervals(
            fasta=fasta,
            positives=train_pos,
            all_peak_intervals=all_peak_intervals,
            interval_length=args.interval_length,
            rng=rng,
        )
        test_neg = sample_negative_intervals(
            fasta=fasta,
            positives=test_pos,
            all_peak_intervals=all_peak_intervals,
            interval_length=args.interval_length,
            rng=rng,
        )

    train_rows = [{**r, "split": "train"} for r in (train_pos + train_neg)]
    test_rows = [{**r, "split": "test"} for r in (test_pos + test_neg)]
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "test.csv", test_rows)
    print(
        f"Saved train.csv ({len(train_rows)} rows) and test.csv ({len(test_rows)} rows) to {output_dir}"
    )


if __name__ == "__main__":
    main()
