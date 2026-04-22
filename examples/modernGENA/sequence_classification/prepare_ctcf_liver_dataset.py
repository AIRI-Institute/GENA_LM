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
        description="Prepare train/valid CSV for CTCF liver sequence classification."
    )
    parser.add_argument("--fasta_path", type=str, required=True, help="Path to genome FASTA")
    parser.add_argument("--peaks_bed", type=str, required=True, help="Path to ENCODE BED")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for train.csv and valid.csv",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=10000,
        help="Window size around positive/negative centers (default: 10kb)",
    )
    parser.add_argument(
        "--valid_chromosomes",
        type=str,
        default="chr21",
        help="Comma separated chromosome list for validation split",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Cap positives per split for smoke tests (0 = no cap)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for split/sampling"
    )
    return parser.parse_args()


def normalize_chrom(chrom: str) -> str:
    if chrom.startswith("chr"):
        return chrom
    return f"chr{chrom}"


def load_peaks(bed_path: Path) -> List[PeakRecord]:
    peaks: List[PeakRecord] = []
    with bed_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = normalize_chrom(parts[0])
            start = int(parts[1])
            end = int(parts[2])
            summit_offset = -1
            if len(parts) >= 10:
                try:
                    summit_offset = int(parts[9])
                except ValueError:
                    summit_offset = -1
            if summit_offset >= 0:
                center = start + summit_offset
            else:
                center = (start + end) // 2
            peaks.append((chrom, start, end, center))
    return peaks


def overlaps(intervals: Sequence[Interval], start: int, end: int) -> bool:
    if not intervals:
        return False
    starts = [iv[0] for iv in intervals]
    idx = bisect.bisect_left(starts, start)
    check_idx = [idx - 1, idx]
    for i in check_idx:
        if 0 <= i < len(intervals):
            iv_start, iv_end = intervals[i]
            if start < iv_end and end > iv_start:
                return True
    return False


def build_positive_rows(
    fasta: pysam.FastaFile,
    peaks: Sequence[PeakRecord],
    sequence_length: int,
) -> Tuple[List[dict], Dict[str, List[Interval]]]:
    half_window = sequence_length // 2
    rows: List[dict] = []
    intervals_by_chrom: Dict[str, List[Interval]] = defaultdict(list)
    known_chroms = set(fasta.references)

    for chrom, _, _, center in peaks:
        if chrom not in known_chroms:
            continue
        chrom_len = fasta.get_reference_length(chrom)
        start = max(0, center - half_window)
        end = start + sequence_length
        if end > chrom_len:
            end = chrom_len
            start = max(0, end - sequence_length)
        if end - start < sequence_length:
            continue
        seq = fasta.fetch(chrom, start, end).upper()
        if len(seq) != sequence_length or "N" in seq:
            continue
        rows.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "sequence": seq,
                "label": 1,
            }
        )
        intervals_by_chrom[chrom].append((start, end))

    for chrom in intervals_by_chrom:
        intervals_by_chrom[chrom].sort(key=lambda x: x[0])
    return rows, intervals_by_chrom


def build_peak_intervals(peaks: Sequence[PeakRecord]) -> Dict[str, List[Interval]]:
    intervals_by_chrom: Dict[str, List[Interval]] = defaultdict(list)
    for chrom, start, end, _ in peaks:
        intervals_by_chrom[chrom].append((start, end))
    for chrom in intervals_by_chrom:
        intervals_by_chrom[chrom].sort(key=lambda x: x[0])
    return intervals_by_chrom


def sample_negative_rows(
    fasta: pysam.FastaFile,
    positives: Sequence[dict],
    exclusion_intervals_by_chrom: Dict[str, List[Interval]],
    sequence_length: int,
    rng: random.Random,
) -> List[dict]:
    negatives: List[dict] = []
    by_chrom_positive_counts: Dict[str, int] = defaultdict(int)
    for row in positives:
        by_chrom_positive_counts[row["chrom"]] += 1

    for chrom, n_samples in by_chrom_positive_counts.items():
        chrom_len = fasta.get_reference_length(chrom)
        if chrom_len <= sequence_length:
            continue
        exclusion_intervals = exclusion_intervals_by_chrom.get(chrom, [])
        sampled = 0
        max_tries = max(1000, n_samples * 300)
        tries = 0
        while sampled < n_samples and tries < max_tries:
            tries += 1
            start = rng.randint(0, chrom_len - sequence_length)
            end = start + sequence_length
            # Negative windows must not intersect any known CTCF peak region.
            if overlaps(exclusion_intervals, start, end):
                continue
            seq = fasta.fetch(chrom, start, end).upper()
            if len(seq) != sequence_length or "N" in seq:
                continue
            negatives.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "sequence": seq,
                    "label": 0,
                }
            )
            sampled += 1
    return negatives


def write_csv(path: Path, rows: Sequence[dict]):
    fields = ["chrom", "start", "end", "sequence", "label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = Path(args.fasta_path).expanduser().absolute()
    bed_path = Path(args.peaks_bed).expanduser().absolute()
    valid_chroms = {normalize_chrom(ch.strip()) for ch in args.valid_chromosomes.split(",") if ch.strip()}

    peaks = load_peaks(bed_path)
    if not peaks:
        raise RuntimeError(f"No peaks parsed from {bed_path}")

    with pysam.FastaFile(str(fasta_path)) as fasta:
        train_peaks = [p for p in peaks if p[0] not in valid_chroms]
        valid_peaks = [p for p in peaks if p[0] in valid_chroms]
        if not valid_peaks:
            raise RuntimeError(
                f"No peaks found for validation chromosomes: {sorted(valid_chroms)}"
            )

        train_rows_pos, _ = build_positive_rows(
            fasta=fasta,
            peaks=train_peaks,
            sequence_length=args.sequence_length,
        )
        valid_rows_pos, _ = build_positive_rows(
            fasta=fasta,
            peaks=valid_peaks,
            sequence_length=args.sequence_length,
        )
        train_peak_intervals = build_peak_intervals(train_peaks)
        valid_peak_intervals = build_peak_intervals(valid_peaks)

        if args.max_samples and args.max_samples > 0:
            rng.shuffle(train_rows_pos)
            rng.shuffle(valid_rows_pos)
            train_rows_pos = train_rows_pos[: args.max_samples]
            valid_rows_pos = valid_rows_pos[: args.max_samples]
            # Keep peak exclusions from full split to avoid sampling near any known peak.
            train_intervals = {
                chrom: intervals[:]
                for chrom, intervals in train_peak_intervals.items()
            }
            valid_intervals = {
                chrom: intervals[:]
                for chrom, intervals in valid_peak_intervals.items()
            }
        else:
            train_intervals = train_peak_intervals
            valid_intervals = valid_peak_intervals

        train_rows_neg = sample_negative_rows(
            fasta=fasta,
            positives=train_rows_pos,
            exclusion_intervals_by_chrom=train_intervals,
            sequence_length=args.sequence_length,
            rng=rng,
        )
        valid_rows_neg = sample_negative_rows(
            fasta=fasta,
            positives=valid_rows_pos,
            exclusion_intervals_by_chrom=valid_intervals,
            sequence_length=args.sequence_length,
            rng=rng,
        )

    train_rows = train_rows_pos + train_rows_neg
    valid_rows = valid_rows_pos + valid_rows_neg
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    write_csv(output_dir / "train.csv", train_rows)
    write_csv(output_dir / "valid.csv", valid_rows)

    print(
        f"Saved train.csv ({len(train_rows)} rows) and valid.csv ({len(valid_rows)} rows) to {output_dir}"
    )


if __name__ == "__main__":
    main()
