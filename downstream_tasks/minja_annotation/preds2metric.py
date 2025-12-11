#!/usr/bin/env python3
"""
Script to process bigWig files and compute TSS/PolyA metrics and per-base ROC-AUC.
Reads bigWig files, applies threshold, combines strands, and generates overlap plots.
Also computes per-base ROC–AUC and F1-vs-threshold curves, printing the best F1 and its
threshold per class (TSS+/TSS-/PolyA+/PolyA-, and Intragenic+/Intragenic- if available).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import pyBigWig as bw
import matplotlib.pyplot as plt

# Import the metric computation functions
sys.path.append('../annotation/compute_metric')
from tss_polya_metric_v2 import compute_metrics


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process bigWig files and compute TSS/PolyA metrics')
    parser.add_argument('--bigwig_path', type=str, required=True,
                        help='Path to directory containing bigWig files')
    parser.add_argument('--gt_path', type=str,
                        default='../annotation/compute_metric/NC_060944.1.tsv',
                        help='Path to ground truth TSV file')
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help='Threshold for converting bigWig values to binary predictions')
    parser.add_argument('--max_k', type=int,
                        default=50,
                        help='Maximum number of peaks to consider for overlap calculation')
    parser.add_argument('--postprocess', action='store_true', default=False,
                        help='Apply postprocessing to keep only highest signal bp in continuous regions')
    parser.add_argument('--min_dist', type=int, default=0,
                        help='Minimum distance: fill gaps shorter than min_dist with 1s before postprocessing')
    parser.add_argument('--strand', type=str, default="+",
                        help='Strand to process: "+", "-", "one_direction", or "both" (affects overlap plots only)')
    parser.add_argument('--chromosome', type=str, default="NC_060944.1",
                        help='Chromosome name as present in the bigWig (default: NC_060944.1)')
    return parser.parse_args()


def validate_bigwig_files(bigwig_path):
    """Validate that all 4 required bigWig files exist (TSS/PolyA for + and -)."""
    required_files = []
    for class_name in ["tss", "polya"]:
        for strand in ["+", "-"]:
            filename = f"{class_name}_{strand}.bw"
            required_files.append(filename)

    missing_files = [fn for fn in required_files if not os.path.exists(os.path.join(bigwig_path, fn))]
    if missing_files:
        raise FileNotFoundError(f"Missing bigWig files: {missing_files}")

    print(f"✓ Found all required bigWig files in {bigwig_path}")
    return required_files


def read_bigwig_chromosome(bigwig_path, filename, chromosome="NC_060944.1"):
    """Read all values from a specific chromosome in a bigWig file."""
    filepath = os.path.join(bigwig_path, filename)

    with bw.open(filepath) as bw_file:
        # Get chromosome length
        chroms = bw_file.chroms()
        if chromosome not in chroms:
            raise KeyError(f"Chromosome {chromosome} not found in {filename}")
        chrom_length = chroms[chromosome]
        print(f"Reading {filename}: chromosome {chromosome} (length: {chrom_length})")

        # Read all values for the chromosome
        values = bw_file.values(chromosome, 0, chrom_length)

        # Convert None values to 0 (pyBigWig returns None for missing values)
        values = [0.0 if v is None else v for v in values]

    return np.array(values, dtype=np.float32)


def apply_threshold(values, threshold):
    """Apply threshold to convert values to binary 0/1 predictions."""
    return (values >= threshold).astype(int)


def combine_strands(tss_plus, tss_minus, polya_plus, polya_minus):
    """Combine strand predictions using max() operation."""
    tss_combined = np.maximum(tss_plus, tss_minus)
    polya_combined = np.maximum(polya_plus, polya_minus)
    return tss_combined, polya_combined


def merge_rev_comp(signal_plus, signal_minus):
    """Merge plus and minus strands of a signal."""
    return np.mean([signal_plus, signal_minus], axis=0)


def postprocess_signal(binary_signal, original_signal, min_dist=0):
    """
    Postprocess binary signal to keep only the highest signal bp in continuous regions.
    First fills gaps shorter than min_dist with 1s, then applies peak-finding logic.
    Uses vectorized operations for faster processing.
    """
    if len(binary_signal) != len(original_signal):
        raise ValueError("Binary signal and original signal must have the same length")

    # Convert to numpy arrays for vectorized operations
    binary = np.array(binary_signal, dtype=int)
    original = np.array(original_signal, dtype=float)

    # Step 1: Fill gaps shorter than min_dist with 1s
    if min_dist > 0:
        binary = fill_short_gaps(binary, min_dist)

    # Step 2: Find region starts and ends using vectorized operations
    # Region starts: where binary goes from 0 to 1
    shifted_binary = np.concatenate([[0], binary[:-1]])  # Shift right by 1
    region_starts = np.where((binary == 1) & (shifted_binary == 0))[0]

    # Region ends: where binary goes from 1 to 0
    shifted_binary_end = np.concatenate([binary[1:], [0]])  # Shift left by 1
    region_ends = np.where((binary == 1) & (shifted_binary_end == 0))[0]

    # Handle case where region extends to the end of the array
    if len(region_starts) > len(region_ends):
        region_ends = np.concatenate([region_ends, [len(binary) - 1]])

    # Initialize output array
    postprocessed = np.zeros_like(binary)

    if len(region_starts) > 0:
        print("Mean positive region length: ", np.mean(region_ends - region_starts + 1))
    else:
        print("Mean positive region length: 0")

    # Step 3: Process each region - keep only highest signal bp
    for start, end in zip(region_starts, region_ends):
        # Find the position with highest original signal in this region
        region_original = original[start:end + 1]
        max_idx = np.argmax(region_original)
        max_position = start + max_idx
        postprocessed[max_position] = 1

    print("Sum of postprocessed: ", np.sum(postprocessed), " sum of the original: ", np.sum(binary))
    return postprocessed


def fill_short_gaps(binary_signal, min_dist):
    """
    Fill gaps (sequences of 0s) shorter than min_dist with 1s.

    Args:
        binary_signal: Binary array (0/1)
        min_dist: Minimum distance - gaps shorter than this are filled

    Returns:
        Binary array with short gaps filled
    """
    if min_dist <= 0:
        return binary_signal

    filled = binary_signal.copy()

    # Find gap starts and ends using vectorized operations
    # Gap starts: where signal goes from 1 to 0
    shifted_binary = np.concatenate([[1], binary_signal[:-1]])  # Shift right by 1
    gap_starts = np.where((binary_signal == 0) & (shifted_binary == 1))[0]

    # Gap ends: where signal goes from 0 to 1
    shifted_binary_end = np.concatenate([binary_signal[1:], [1]])  # Shift left by 1
    gap_ends = np.where((binary_signal == 0) & (shifted_binary_end == 1))[0]

    # Handle case where gap extends to the end of the array
    if len(gap_starts) > len(gap_ends):
        gap_ends = np.concatenate([gap_ends, [len(binary_signal) - 1]])

    # Fill gaps shorter than min_dist
    for start, end in zip(gap_starts, gap_ends):
        gap_length = end - start + 1
        if gap_length <= min_dist:
            filled[start:end + 1] = 1

    return filled


def compute_metrics_and_plot(predictions, gt_path, label, output_path, max_k=250):
    """Compute overlap metrics and generate plot."""
    print(f"Computing metrics for {label}...")

    # Load ground truth data
    df = pd.read_csv(gt_path, sep='\t')

    results = compute_metrics(predictions, df=df, label=label, output_path=output_path, max_k=max_k)
    return results


# --------------------------- AUC + F1 utilities ---------------------------

def _build_gt_arrays(df: pd.DataFrame, length: int):
    """
    Build per-base binary ground-truth arrays:
      - TSS_plus / TSS_minus: 1 at TSS position for each transcript on that strand
      - PolyA_plus / PolyA_minus: 1 at PolyA position for each transcript on that strand
      - Intragenic_plus / Intragenic_minus: 1 across [min(TSS, PolyA), max(TSS, PolyA)] (union) by strand

    Assumes TSV positions are 1-based. Clips indices to [0, length-1].
    """
    gt = {
        "TSS_plus": np.zeros(length, dtype=np.uint8),
        "TSS_minus": np.zeros(length, dtype=np.uint8),
        "PolyA_plus": np.zeros(length, dtype=np.uint8),
        "PolyA_minus": np.zeros(length, dtype=np.uint8),
        "Intragenic_plus": np.zeros(length, dtype=np.uint8),
        "Intragenic_minus": np.zeros(length, dtype=np.uint8),
    }

    local = df.copy()
    for col in ["TSS", "PolyA"]:
        local[col] = pd.to_numeric(local[col], errors="coerce")
    local = local.dropna(subset=["TSS", "PolyA", "strand"])

    for row in local.itertuples(index=False):
        try:
            tss = int(getattr(row, "TSS"))
            polya = int(getattr(row, "PolyA"))
            strand = getattr(row, "strand")
        except Exception:
            continue

        # Convert to 0-based indices
        tss_idx = max(0, min(length - 1, tss - 1))
        polya_idx = max(0, min(length - 1, polya - 1))

        if strand == "+":
            gt["TSS_plus"][tss_idx] = 1
            gt["PolyA_plus"][polya_idx] = 1
            start = min(tss_idx, polya_idx)
            end = max(tss_idx, polya_idx)
            gt["Intragenic_plus"][start:end + 1] = 1
        elif strand == "-":
            gt["TSS_minus"][tss_idx] = 1
            gt["PolyA_minus"][polya_idx] = 1
            start = min(tss_idx, polya_idx)
            end = max(tss_idx, polya_idx)
            gt["Intragenic_minus"][start:end + 1] = 1

    return gt


def _roc_auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC-AUC via rank-sum (handles ties). Does not require sklearn.
    AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    """
    y_true = y_true.astype(np.uint8)
    y_score = y_score.astype(np.float64)

    # Filter NaNs if any
    mask = ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    n_pos = int(y_true.sum())
    n = y_true.shape[0]
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")  # stable for tie groups
    s_scores = y_score[order]
    s_labels = y_true[order]

    # Average ranks for ties; ranks are 1..n
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s_scores[j] == s_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j

    sum_ranks_pos = (ranks * s_labels).sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Try sklearn if available; otherwise use tie-aware rank-sum AUC."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true.astype(np.uint8), y_score.astype(np.float64)))
    except Exception:
        return _roc_auc_rank(y_true, y_score)


def compute_best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 200):
    """
    Compute best F1 and corresponding threshold by scanning a grid over score range.
    Returns (best_thresh, best_f1, thresholds, f1_values). If F1 is undefined (no positives
    or no negatives), returns (None, NaN, None, None).
    """
    y_true = y_true.astype(np.uint8)
    y_score = y_score.astype(np.float64)

    mask = ~np.isnan(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    n_pos = int(y_true.sum())
    n = y_true.size
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0 or n == 0:
        return None, float("nan"), None, None

    s_min = float(y_score.min())
    s_max = float(y_score.max())

    if s_min == s_max:
        thresholds = np.array([s_min], dtype=np.float64)
    else:
        thresholds = np.linspace(s_min, s_max, num_thresholds)

    f1_values = np.zeros_like(thresholds, dtype=np.float64)

    for i, thr in enumerate(thresholds):
        y_pred = (y_score >= thr)
        tp = np.logical_and(y_pred, y_true == 1).sum()
        fp = np.logical_and(y_pred, y_true == 0).sum()
        fn = np.logical_and(~y_pred, y_true == 1).sum()

        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * precision * recall / (precision + recall)
        f1_values[i] = f1

    best_idx = int(np.argmax(f1_values))
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1_values[best_idx])

    return best_thr, best_f1, thresholds, f1_values


def plot_f1_vs_threshold(thresholds, f1_values, best_thr, best_f1, title, output_path):
    """Plot F1 vs threshold and mark the best threshold."""
    plt.figure()
    plt.plot(thresholds, f1_values, label="F1")
    plt.axvline(best_thr, linestyle="--", label=f"best thr={best_thr:.3f}")
    plt.scatter([best_thr], [best_f1], zorder=5)
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title(f"{title} – F1 vs threshold\n(best F1={best_f1:.3f} at thr={best_thr:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_and_print_auc(bigwig_path: str, gt_path: str, chromosome: str):
    """
    Compute per-base ROC-AUC over the entire chromosome for:
      - TSS_plus, TSS_minus, PolyA_plus, PolyA_minus
      - Intragenic_plus, Intragenic_minus (if both intragenic bigWigs exist)

    Also compute, per class, the F1-vs-threshold curve and:
      - print best F1 and its threshold
      - plot F1 vs threshold with the best threshold marked
    """
    # Always read the strand-specific *non-revcomp* files for AUC and F1
    tss_plus_scores = read_bigwig_chromosome(bigwig_path, "tss_+.bw", chromosome)
    tss_minus_scores = read_bigwig_chromosome(bigwig_path, "tss_-.bw", chromosome)
    polya_plus_scores = read_bigwig_chromosome(bigwig_path, "polya_+.bw", chromosome)
    polya_minus_scores = read_bigwig_chromosome(bigwig_path, "polya_-.bw", chromosome)

    length = len(tss_plus_scores)
    assert all(len(x) == length for x in [tss_minus_scores, polya_plus_scores, polya_minus_scores]), \
        "All bigWig arrays must be same length"

    intragenic_plus_path = os.path.join(bigwig_path, "intragenic_+.bw")
    intragenic_minus_path = os.path.join(bigwig_path, "intragenic_-.bw")
    have_intragenic = os.path.exists(intragenic_plus_path) and os.path.exists(intragenic_minus_path)

    if have_intragenic:
        intragenic_plus_scores = read_bigwig_chromosome(bigwig_path, "intragenic_+.bw", chromosome)
        intragenic_minus_scores = read_bigwig_chromosome(bigwig_path, "intragenic_-.bw", chromosome)
        assert len(intragenic_plus_scores) == length and len(intragenic_minus_scores) == length, \
            "Intragenic bigWigs must match chromosome length"

    # Ground truth arrays
    df = pd.read_csv(gt_path, sep='\t')
    gt = _build_gt_arrays(df, length)

    print("\n=== Per-base ROC–AUC (entire chromosome) ===")
    aucs = {}

    classes = [
        ("TSS (+)", "TSS_plus", tss_plus_scores, "TSS_plus"),
        ("TSS (-)", "TSS_minus", tss_minus_scores, "TSS_minus"),
        ("PolyA (+)", "PolyA_plus", polya_plus_scores, "PolyA_plus"),
        ("PolyA (-)", "PolyA_minus", polya_minus_scores, "PolyA_minus"),
    ]
    if have_intragenic:
        classes.extend([
            ("Intragenic (+)", "Intragenic_plus", intragenic_plus_scores, "Intragenic_plus"),
            ("Intragenic (-)", "Intragenic_minus", intragenic_minus_scores, "Intragenic_minus"),
        ])

    # AUC printing
    for label, key, scores, short in classes:
        auc = _roc_auc(gt[key], scores)
        aucs[label] = auc
        if np.isnan(auc):
            print(f"{label}: N/A (no positives or negatives)")
        else:
            print(f"{label}: {auc:.6f}")

    # F1 and best threshold printing + plotting
    print("\n=== Best F1 and threshold (per class) ===")
    for label, key, scores, short in classes:
        best_thr, best_f1, thresholds, f1_values = compute_best_f1_threshold(gt[key], scores)
        if best_thr is None or thresholds is None:
            print(f"{label}: F1 N/A (no positives or negatives)")
            continue

        print(f"{label}: best F1 = {best_f1:.6f} at threshold = {best_thr:.6f}")

        fname = f"F1_vs_threshold_{short}.png"
        out_path = os.path.join(bigwig_path, fname)
        plot_f1_vs_threshold(thresholds, f1_values, best_thr, best_f1, label, out_path)
        print(f"Saved F1-vs-threshold plot for {label} to: {out_path}")

    if not have_intragenic:
        print("Intragenic: skipped for AUC/F1 (intragenic_+.bw and/or intragenic_-.bw not found)")

    return aucs


# --------------------------- main ---------------------------

def main():
    """Main function."""
    args = parse_arguments()

    # Validate inputs
    if not os.path.exists(args.bigwig_path):
        raise FileNotFoundError(f"BigWig path does not exist: {args.bigwig_path}")

    if not os.path.exists(args.gt_path):
        raise FileNotFoundError(f"GT file does not exist: {args.gt_path}")

    # Validate bigWig files
    _ = validate_bigwig_files(args.bigwig_path)

    # Set output directory (same as bigWig directory)
    output_dir = args.bigwig_path

    print(f"Processing with threshold: {args.threshold}")
    print(f"Postprocessing enabled: {args.postprocess}")
    if args.postprocess:
        print(f"Minimum distance for gap filling: {args.min_dist}")
    print(f"Output directory: {output_dir}")

    # Read bigWig files for overlap-plot predictions
    print("\nReading bigWig files (for overlap metrics)...")

    if args.strand == "+":  # genes on + and - as original
        tss_plus = read_bigwig_chromosome(args.bigwig_path, "tss_+.bw", args.chromosome)
        tss_minus = read_bigwig_chromosome(args.bigwig_path, "tss_-.bw", args.chromosome)
        polya_plus = read_bigwig_chromosome(args.bigwig_path, "polya_+.bw", args.chromosome)
        polya_minus = read_bigwig_chromosome(args.bigwig_path, "polya_-.bw", args.chromosome)
    elif args.strand == "one_direction":
        tss_plus = read_bigwig_chromosome(args.bigwig_path, "tss_+.bw", args.chromosome)
        tss_minus = read_bigwig_chromosome(args.bigwig_path, "tss_-rev_comp_.bw", args.chromosome)
        polya_plus = read_bigwig_chromosome(args.bigwig_path, "polya_+.bw", args.chromosome)
        polya_minus = read_bigwig_chromosome(args.bigwig_path, "polya_-rev_comp_.bw", args.chromosome)
    elif args.strand == "-":
        tss_plus = read_bigwig_chromosome(args.bigwig_path, "tss_-rev_comp_.bw", args.chromosome)
        tss_minus = read_bigwig_chromosome(args.bigwig_path, "tss_+rev_comp_.bw", args.chromosome)
        polya_plus = read_bigwig_chromosome(args.bigwig_path, "polya_-rev_comp_.bw", args.chromosome)
        polya_minus = read_bigwig_chromosome(args.bigwig_path, "polya_+rev_comp_.bw", args.chromosome)
    elif args.strand == "both":
        tss_plus = merge_rev_comp(
            read_bigwig_chromosome(args.bigwig_path, "tss_+.bw", args.chromosome),
            read_bigwig_chromosome(args.bigwig_path, "tss_-rev_comp_.bw", args.chromosome)
        )
        tss_minus = merge_rev_comp(
            read_bigwig_chromosome(args.bigwig_path, "tss_-.bw", args.chromosome),
            read_bigwig_chromosome(args.bigwig_path, "tss_+rev_comp_.bw", args.chromosome)
        )
        polya_plus = merge_rev_comp(
            read_bigwig_chromosome(args.bigwig_path, "polya_+.bw", args.chromosome),
            read_bigwig_chromosome(args.bigwig_path, "polya_-rev_comp_.bw", args.chromosome)
        )
        polya_minus = merge_rev_comp(
            read_bigwig_chromosome(args.bigwig_path, "polya_-.bw", args.chromosome),
            read_bigwig_chromosome(args.bigwig_path, "polya_+rev_comp_.bw", args.chromosome)
        )
    else:
        raise ValueError(f"Invalid strand: {args.strand}")

    # Apply threshold to convert to binary predictions
    print(f"\nApplying threshold {args.threshold}...")
    tss_plus_binary = apply_threshold(tss_plus, args.threshold)
    tss_minus_binary = apply_threshold(tss_minus, args.threshold)
    polya_plus_binary = apply_threshold(polya_plus, args.threshold)
    polya_minus_binary = apply_threshold(polya_minus, args.threshold)

    # Apply postprocessing if requested
    if args.postprocess:
        print("Applying postprocessing to keep only highest signal bp in continuous regions...")
        if args.min_dist > 0:
            print(f"Filling gaps shorter than {args.min_dist} bp with 1s...")
        tss_plus_binary = postprocess_signal(tss_plus_binary, tss_plus, args.min_dist)
        tss_minus_binary = postprocess_signal(tss_minus_binary, tss_minus, args.min_dist)
        polya_plus_binary = postprocess_signal(polya_plus_binary, polya_plus, args.min_dist)
        polya_minus_binary = postprocess_signal(polya_minus_binary, polya_minus, args.min_dist)

    # Combine strands using max()
    print("Combining strands...")
    tss_combined, polya_combined = combine_strands(
        tss_plus_binary, tss_minus_binary,
        polya_plus_binary, polya_minus_binary
    )

    print(f"TSS combined: {np.sum(tss_combined)} positive predictions out of {len(tss_combined)}")
    print(f"PolyA combined: {np.sum(polya_combined)} positive predictions out of {len(polya_combined)}")

    # Generate output filenames with threshold and postprocessing flags
    postprocess_suffix = "_postprocessed" if args.postprocess else ""
    min_dist_suffix = f"_mindist_{args.min_dist}" if args.postprocess and args.min_dist > 0 else ""
    strand_suffix = f"_strand_{args.strand}" if args.strand != "+" else ""
    tss_output = os.path.join(output_dir, f"TSS_threshold_{args.threshold}_max_k_{args.max_k}{postprocess_suffix}{min_dist_suffix}{strand_suffix}.png")
    polya_output = os.path.join(output_dir, f"PolyA_threshold_{args.threshold}_max_k_{args.max_k}{postprocess_suffix}{min_dist_suffix}{strand_suffix}.png")

    # Compute metrics and generate plots
    print("\nComputing TSS metrics...")
    results_tss = compute_metrics_and_plot(
        tss_combined, args.gt_path, "TSS", tss_output,
        max_k=args.max_k
    )

    print("\nComputing PolyA metrics...")
    results_polya = compute_metrics_and_plot(
        polya_combined, args.gt_path, "PolyA", polya_output,
        max_k=args.max_k
    )

    # Write the overlap metrics to a text file
    metrics_path = os.path.join(
        output_dir,
        f"metrics_threshold_{args.threshold}_max_k_{args.max_k}{postprocess_suffix}{min_dist_suffix}{strand_suffix}.txt"
    )
    with open(metrics_path, "w") as f:
        # TSS O (k=50) TSS NO (k=50) PolyA O (k=50) PolyA NO (k=50) TSS O (k=250) ...
        for k in [50, 250]:
            for r in [results_tss, results_polya]:
                for t in ["O", "NO"]:
                    f.write(str(r[t + "_" + str(k)]) + "\t")
        f.write("\n")

    print(f"\nWrote overlap metrics to: {metrics_path}")

    # NEW: Compute and print AUCs and best-F1 thresholds per class
    print("\nComputing per-base ROC–AUC and best F1/threshold for strand-specific classes...")
    aucs = compute_and_print_auc(args.bigwig_path, args.gt_path, args.chromosome)

    print("\n✓ Processing complete!")
    print(f"✓ TSS plot saved to: {tss_output}")
    print(f"✓ PolyA plot saved to: {polya_output}")


if __name__ == "__main__":
    main()
