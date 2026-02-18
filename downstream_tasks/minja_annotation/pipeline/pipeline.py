#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pyBigWig as bw

from tqdm import tqdm
from scipy.signal import find_peaks
from typing import List, Tuple, Optional, Dict, Union

# ============================ CONFIG ============================
chrom = 'NC_060944.1'
chrom_len = 66210255
# files_path = '/disk/10tb/home/shmelev/GENA_LM/runs/annotation/MLGENX_modernGENA_rc_shift_mRNA_and_lncRNA_middle_pretrain_full_BCE/checkpoint-22750/eval/T2T-CHM13v2/NC_060944.1'
files_path = '/disk/10tb/home/shmelev/GENA_LM/runs/annotation/copy_from_hpc_hse/MLGENX_modernGENA_rc_shift_middle_pretrain_shawerma_4_classes_1024/checkpoint-147500/eval/T2T-CHM13v2/NC_060944.1'

#INPUT_PATH = '/home/jovyan/shares/SR003.nfs1/dnalm/downstream_tasks/annotation/peak_finding/modernGENA_SHAWERMA_HSE_HPC_rc_shift_lncRNA_andmRNA_full_BCE_middle_pretrain.npy' # '/home/jovyan/shmelev/dnalm/downstream_tasks/annotation/peak_finding/modernGENA_rc_shift_lncRNA_andmRNA_full_BCE_middle_pretrain.npy'
#X = None
# OUT_PATH = '/home/jovyan/shares/SR003.nfs1/dnalm/downstream_tasks/annotation/peak_finding/called_peaks_modernGENA_SHAWERMA_HSE_HPC_rc_shift_lncRNA_andmRNA_full_BCE_middle_pretrain.npy' # '/home/jovyan/shmelev/dnalm/downstream_tasks/annotation/peak_finding/called_peaks_modernGENA_rc_shift_lncRNA_andmRNA_full_BCE_middle_pretrain_stranded_tss_polya_only_lower_thresholds.npy'

LP_FRAC   = 0.05
PK_PROM   = 0.1
PK_DIST   = 50
PK_HEIGHT = None

BW_PLUS          = "intragenic_+.bw"
BW_MINUS         = "intragenic_-.bw"
BW_PLUS_RC       = "intragenic_+rev_comp_.bw"
BW_MINUS_RC      = "intragenic_-rev_comp_.bw"

# args_bw_dir = "/disk/10tb/home/shmelev/GENA_LM/runs/annotation/MLGENX_modernGENA_rc_shift_middle_pretrain_6_classes_with_intragenic_8192/checkpoint-15750/eval/T2T-CHM13v2/NC_060944.1"
args_bw_dir = '/disk/10tb/home/shmelev/GENA_LM/runs/annotation/copy_from_hpc_hse/MLGENX_modernGENA_rc_shift_middle_pretrain_shawerma_6_classes_1024/checkpoint-268750/eval/T2T-CHM13v2/NC_060944.1'
args_prob_threshold = float(0.5)
args_zero_fraction_drop_threshold = float(0.01)
args_bed_out = f"/disk/10tb/home/shmelev/artem_from_protein_allocation/MLGenX2026/moderngena_shawerma_thr_{PK_PROM}.bed"
# args_h5_out = f"/disk/10tb/home/shmelev/artem_from_protein_allocation/MLGenX2026/moderngena_shawerma_thr_{PK_PROM}.hdf5"
# ================================================================


def combine_strands(tss_plus, tss_minus, polya_plus, polya_minus):
    tss_combined = np.maximum(tss_plus, tss_minus)
    polya_combined = np.maximum(polya_plus, polya_minus)
    return tss_combined, polya_combined


def merge_rev_comp(signal_plus, signal_minus):
    return np.mean([signal_plus, signal_minus], axis=0)


def read_bigwig(bigwig_path, filename, chromosome="NC_060944.1"):
    filepath = os.path.join(bigwig_path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"BigWig file not found: {filepath}")

    with bw.open(filepath) as bw_file:
        if chromosome not in bw_file.chroms():
            raise ValueError(f"Chromosome {chromosome} not found in {filename}")

        chrom_length = bw_file.chroms()[chromosome]
        print(f"Reading {filename}: chromosome {chromosome} (length: {chrom_length})")

        values = bw_file.values(chromosome, 0, chrom_length)
        values = [0.0 if v is None else v for v in values]

    return np.array(values, dtype=np.float32)


def prepare_preds_for_peaks(files_path, strand):
    if strand == "both":
        data = {}
        for label in ['tss', 'polya']:
            for file_strand in ['+', '-']:
                file_name = f"{label}_{file_strand}"
                data[file_name] = read_bigwig(files_path, f"{file_name}.bw")

                rc_file_name = f"{label}_{file_strand}rev_comp_"
                data[rc_file_name] = read_bigwig(files_path, f"{rc_file_name}.bw")

        for name in data.keys():
            print(f"{name}.max = {np.max(data[name])}")

        tss_plus = merge_rev_comp(data["tss_+"], data["tss_-rev_comp_"])
        tss_minus = merge_rev_comp(data["tss_-"], data["tss_+rev_comp_"])

        polya_plus = merge_rev_comp(data["polya_+"], data["polya_-rev_comp_"])
        polya_minus = merge_rev_comp(data["polya_-"], data["polya_+rev_comp_"])

        tss_combined, polya_combined = combine_strands(tss_plus, tss_minus, polya_plus, polya_minus)

    else:
        data = {}
        for label in ['tss', 'polya']:
            for file_strand in ['+', '-']:
                file_name = f"{label}_{file_strand}"
                data[file_name] = read_bigwig(files_path, f"{file_name}.bw")

        tss_plus = data["tss_+"]
        tss_minus = data["tss_-"]

        polya_plus = data["polya_+"]
        polya_minus = data["polya_-"]

        tss_combined, polya_combined = combine_strands(tss_plus, tss_minus, polya_plus, polya_minus)

    return tss_combined, polya_combined


def prepare_preds_for_peaks_onlyRC(files_path):
    data = {}
    for label in ['tss', 'polya']:
        for file_strand in ['+', '-']:
            file_name = f"{label}_{file_strand}"
            data[file_name] = read_bigwig(files_path, f"{file_name}.bw")

            rc_file_name = f"{label}_{file_strand}rev_comp_"
            data[rc_file_name] = read_bigwig(files_path, f"{rc_file_name}.bw")

    for name in data.keys():
        print(f"{name}.max = {np.max(data[name])}")

    tss_plus = merge_rev_comp(data["tss_+"], data["tss_-rev_comp_"])
    tss_minus = merge_rev_comp(data["tss_-"], data["tss_+rev_comp_"])

    polya_plus = merge_rev_comp(data["polya_+"], data["polya_-rev_comp_"])
    polya_minus = merge_rev_comp(data["polya_-"], data["polya_+rev_comp_"])

    return tss_plus, polya_plus, tss_minus, polya_minus


tss_plus, polya_plus, tss_minus, polya_minus = prepare_preds_for_peaks_onlyRC(files_path)

X = np.array([
    tss_plus,
    polya_plus,
    tss_minus,
    polya_minus
])


def fft_lowpass(x: np.ndarray, frac: float) -> np.ndarray:
    Xf = np.fft.rfft(x)
    k = int(np.clip(frac, 0.0, 1.0) * len(Xf))
    if k < 1:
        k = 1
    X_lp = np.zeros_like(Xf)
    X_lp[:k] = Xf[:k]
    y = np.fft.irfft(X_lp, n=len(x))
    return y


def call_peaks_on_segment(x: np.ndarray,
                          lp_frac: float,
                          pk_prom: float,
                          pk_dist: int,
                          pk_height):
    y_lp = fft_lowpass(x, frac=lp_frac)
    idx, _props = find_peaks(y_lp, prominence=pk_prom, distance=pk_dist, height=pk_height)
    return idx, y_lp


def peak_finding(X):
    X = np.nan_to_num(X)
    N = X.shape[1]
    print("Input shape:", X.shape)

    mask = np.zeros((4, N), dtype=np.uint8)

    peak_counts = []
    for r in range(4):
        x = X[r, :].astype(float, copy=False)
        idx_local, _y_lp = call_peaks_on_segment(
            x,
            lp_frac=LP_FRAC,
            pk_prom=PK_PROM,
            pk_dist=PK_DIST,
            pk_height=PK_HEIGHT
        )
        mask[r, idx_local] = 1
        peak_counts.append(len(idx_local))

    return mask


arr = peak_finding(X)


def find_tss_polya_pairs_old(
    arr: np.ndarray,
    chrom_name: str,
    window_size: int = 2_000_000,
    k: Optional[int] = None,
    out_bed_path: Optional[str] = None,
    progress_every: Optional[int] = None,
) -> List[Tuple[int, int]]:
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError("arr must have shape (2, X)")
    X = int(arr.shape[1])

    if window_size > X:
        window_size = X

    tss_idx = np.flatnonzero(arr[0].astype(np.bool_, copy=False))
    polya_idx = np.flatnonzero(arr[1].astype(np.bool_, copy=False))

    tss_idx.sort()
    polya_idx.sort()

    pairs_sign: Dict[Tuple[int, int], str] = {}
    half = window_size // 2

    def compute_window(center: int) -> Tuple[int, int]:
        start = center - half
        end = start + window_size
        if start < 0:
            start = 0
            end = window_size
        if end > X:
            end = X
            start = X - window_size
            if start < 0:
                start = 0
        return start, end

    def choose_k_nearest(seed: int, candidates: np.ndarray) -> np.ndarray:
        if k is None or k <= 0 or candidates.size <= k:
            return candidates
        dist = np.abs(candidates - seed)
        idx = np.argpartition(dist, k - 1)[:k]
        order = np.lexsort((candidates[idx], dist[idx]))
        return candidates[idx][order]

    def scan(
        seeds: np.ndarray,
        targets: np.ndarray,
        seeds_name: str,
        targets_name: str,
        seeds_is_tss: bool,
        targets_is_tss: bool,
    ):
        if progress_every:
            print(f"Scanning {len(seeds):,} {seeds_name} positions against {len(targets):,} {targets_name} positions...")
        for idx, i in enumerate(seeds):
            start_w, end_w = compute_window(int(i))
            left = targets.searchsorted(start_w, side='left')
            right = targets.searchsorted(end_w, side='left')
            if right > left:
                js_window = targets[left:right]
                js_use = choose_k_nearest(i, js_window)
                for j in js_use:
                    a = int(i) if i <= j else int(j)
                    b = int(j) if i <= j else int(i)
                    if a == i:
                        sign = '+' if seeds_is_tss else '-'
                    else:
                        sign = '+' if targets_is_tss else '-'
                    key = (a, b)
                    if key not in pairs_sign:
                        pairs_sign[key] = sign
            if progress_every and (idx + 1) % progress_every == 0:
                print(f"  processed {idx+1:,}/{len(seeds):,}; unique pairs so far: {len(pairs_sign):,}")
        if progress_every:
            print(f"  done. Total unique pairs so far: {len(pairs_sign):,}")

    scan(tss_idx, polya_idx, "TSS", "PolyA", True, False)
    scan(polya_idx, tss_idx, "PolyA", "TSS", False, True)

    pairs_sorted = sorted(pairs_sign.keys(), key=lambda ab: (ab[0], ab[1]))

    if out_bed_path is not None:
        with open(out_bed_path, "w") as bed:
            for a, b in pairs_sorted:
                sign = pairs_sign[(a, b)]
                bed.write(f"{chrom_name}\t{int(a)}\t{int(b)}\t{sign}\n")

    return pairs_sorted


def find_tss_polya_pairs(
    arr: np.ndarray,
    chrom_name: str,
    window_size: int = 2_000_000,
    k: Optional[int] = None,
    out_bed_path: Optional[str] = None,
    progress_every: Optional[int] = None,
) -> List[Tuple[int, int]]:
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError("arr must have shape (4, X) in order: TSS+, PolyA+, TSS-, PolyA-")
    X = int(arr.shape[1])

    if window_size > X:
        window_size = X

    tss_plus_idx = np.flatnonzero(arr[0].astype(np.bool_, copy=False))
    polya_plus_idx = np.flatnonzero(arr[1].astype(np.bool_, copy=False))
    tss_minus_idx = np.flatnonzero(arr[2].astype(np.bool_, copy=False))
    polya_minus_idx = np.flatnonzero(arr[3].astype(np.bool_, copy=False))

    tss_plus_idx.sort()
    polya_plus_idx.sort()
    tss_minus_idx.sort()
    polya_minus_idx.sort()

    pairs_sign: Dict[Tuple[int, int], str] = {}
    half = window_size // 2

    def compute_window(center: int) -> Tuple[int, int]:
        start = center - half
        end = start + window_size
        if start < 0:
            start = 0
            end = window_size
        if end > X:
            end = X
            start = X - window_size
            if start < 0:
                start = 0
        return start, end

    def choose_k_nearest(seed: int, candidates: np.ndarray) -> np.ndarray:
        if k is None or k <= 0 or candidates.size <= k:
            return candidates
        dist = np.abs(candidates - seed)
        idx = np.argpartition(dist, k - 1)[:k]
        order = np.lexsort((candidates[idx], dist[idx]))
        return candidates[idx][order]

    def scan(
        seeds: np.ndarray,
        targets: np.ndarray,
        seeds_name: str,
        targets_name: str,
        seeds_is_plus: bool,
        targets_is_plus: bool,
    ):
        if progress_every:
            print(f"Scanning {len(seeds):,} {seeds_name} positions against {len(targets):,} {targets_name} positions...")
        for idx, i in enumerate(seeds):
            start_w, end_w = compute_window(int(i))
            left = targets.searchsorted(start_w, side='left')
            right = targets.searchsorted(end_w, side='left')
            if right > left:
                js_window = targets[left:right]
                js_use = choose_k_nearest(i, js_window)
                for j in js_use:
                    a, b = (int(i), int(j)) if i <= j else (int(j), int(i))
                    start_is_plus = seeds_is_plus if a == i else targets_is_plus
                    sign = '+' if start_is_plus else '-'
                    pairs_sign.setdefault((a, b), sign)
            if progress_every and (idx + 1) % progress_every == 0:
                print(f"  processed {idx+1:,}/{len(seeds):,}; unique pairs so far: {len(pairs_sign):,}")
        if progress_every:
            print(f"  done. Total unique pairs so far: {len(pairs_sign):,}")

    scan(tss_plus_idx, polya_plus_idx, "TSS+", "PolyA+", True, True)
    scan(polya_plus_idx, tss_plus_idx, "PolyA+", "TSS+", True, True)

    scan(tss_minus_idx, polya_minus_idx, "TSS-", "PolyA-", False, False)
    scan(polya_minus_idx, tss_minus_idx, "PolyA-", "TSS-", False, False)

    pairs_sorted = sorted(pairs_sign.keys(), key=lambda ab: (ab[0], ab[1]))

    if out_bed_path is not None:
        with open(out_bed_path, "w") as bed:
            for a, b in pairs_sorted:
                bed.write(f"{chrom_name}\t{int(a)}\t{int(b)}\t{pairs_sign[(a, b)]}\n")

    return pairs_sorted


def find_tss_polya_pairs_right_left_only(arr, chrom_name, window_size=2_000_000, k=10,
                                        out_bed_path=None, progress_every=None):
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError("arr must have shape (4, X) in order: TSS+, PolyA+, TSS-, PolyA-")
    X = int(arr.shape[1])
    if window_size > X:
        window_size = X

    tss_plus_idx = np.flatnonzero(arr[0].astype(np.bool_, copy=False)); tss_plus_idx.sort()
    polya_plus_idx = np.flatnonzero(arr[1].astype(np.bool_, copy=False)); polya_plus_idx.sort()
    tss_minus_idx = np.flatnonzero(arr[2].astype(np.bool_, copy=False)); tss_minus_idx.sort()
    polya_minus_idx = np.flatnonzero(arr[3].astype(np.bool_, copy=False)); polya_minus_idx.sort()

    pairs_sign = {}

    def choose_k_nearest(seed: int, candidates: np.ndarray) -> np.ndarray:
        if k is None or k <= 0 or candidates.size <= k:
            return candidates
        dist = np.abs(candidates - seed)
        idx = np.argpartition(dist, k - 1)[:k]
        order = np.lexsort((candidates[idx], dist[idx]))
        return candidates[idx][order]

    def scan_tss_to_polya_one_sided(seeds_tss: np.ndarray, targets_polya: np.ndarray,
                                   direction: str, strand_sign: str, label: str):
        if progress_every:
            print(f"Scanning {len(seeds_tss):,} {label} TSS seeds one sided")
        for ii, i in enumerate(seeds_tss):
            i = int(i)

            if direction == "right":
                start_w = i
                end_w = min(X, i + window_size)
                left = targets_polya.searchsorted(start_w, side="left")
                right = targets_polya.searchsorted(end_w, side="left")
            elif direction == "left":
                start_w = max(0, i - window_size)
                end_w = i + 1
                left = targets_polya.searchsorted(start_w, side="left")
                right = targets_polya.searchsorted(end_w, side="left")
            else:
                raise ValueError("direction must be 'right' or 'left'")

            if right > left:
                js_window = targets_polya[left:right]
                js_use = choose_k_nearest(i, js_window)

                for j in js_use:
                    j = int(j)
                    a, b = (i, j) if i <= j else (j, i)
                    pairs_sign.setdefault((a, b), strand_sign)

            if progress_every and (ii + 1) % progress_every == 0:
                print(f"  processed {ii+1:,}/{len(seeds_tss):,}; pairs so far: {len(pairs_sign):,}")

    scan_tss_to_polya_one_sided(
        seeds_tss=tss_plus_idx,
        targets_polya=polya_plus_idx,
        direction="right",
        strand_sign="+",
        label="plus"
    )

    scan_tss_to_polya_one_sided(
        seeds_tss=tss_minus_idx,
        targets_polya=polya_minus_idx,
        direction="left",
        strand_sign="-",
        label="minus"
    )

    pairs_sorted = sorted(pairs_sign.keys(), key=lambda ab: (ab[0], ab[1]))

    final_pairs = []
    for a, b in pairs_sorted:
        final_pairs.append((chrom_name, a, b, pairs_sign[(a, b)], []))

    if out_bed_path is not None:
        with open(out_bed_path, "w") as bed:
            for a, b in pairs_sorted:
                bed.write(f"{chrom_name}\t{a}\t{b}\t{pairs_sign[(a, b)]}\n")

    return final_pairs


#arr = np.load('/home/jovyan/shares/SR003.nfs1/dnalm/downstream_tasks/annotation/peak_finding/called_peaks_modernGENA_SHAWERMA_HSE_HPC_rc_shift_lncRNA_andmRNA_full_BCE_middle_pretrain.npy')
pairs = find_tss_polya_pairs_right_left_only(arr, k=10, chrom_name=chrom, out_bed_path=None, progress_every=1000)
print(f"Wrote {len(pairs)} intervals to file.")


def load_bed_intervals(bed_path: str) -> List[Tuple[str, int, int, str, list]]:
    intervals = []
    skipped = 0

    with open(bed_path, "r") as fh:
        for line in fh:
            if not line.strip():
                continue
            if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                skipped += 1
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                skipped += 1
                continue

            chrom = parts[0]
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                skipped += 1
                continue

            strand = "."
            if len(parts) >= 4:
                s = parts[3].strip()
                strand = s if s in {"+", "-"} else "."

            extra = parts[4:] if len(parts) > 4 else []
            intervals.append((chrom, start, end, strand, extra))

    print(f"[INFO] BED loaded: {len(intervals):,} intervals (skipped {skipped} header/invalid lines).")
    return intervals


def fetch_values(handle, chrom: str, start: int, end: int) -> np.ndarray:
    vals = handle.values(chrom, start, end)
    if vals is None:
        return np.zeros(max(0, end - start), dtype=np.float32)
    vals = np.array([0.0 if v is None else v for v in vals], dtype=np.float32)
    np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return vals


def ensure_chrom_available(handles: Dict[str, bw.pyBigWig], chroms_needed: List[str]):
    for name, handle in handles.items():
        chrom_dict = handle.chroms()
        missing = [c for c in chroms_needed if c not in chrom_dict]
        if missing:
            raise ValueError(f"[FATAL] bigWig '{name}' missing chromosomes: {missing}")
        sample = list(chrom_dict.items())[:3]
        print(f"[INFO] '{name}' bigWig has {len(chrom_dict)} chroms. Example entries: {sample}")


def open_bw_inputs(bw_dir_or_file: str):
    if os.path.isdir(bw_dir_or_file):
        bw_dir = bw_dir_or_file
        paths = {
            "intragenic_+":          os.path.join(bw_dir, BW_PLUS),
            "intragenic_-":          os.path.join(bw_dir, BW_MINUS),
            "intragenic_+rev_comp_": os.path.join(bw_dir, BW_PLUS_RC),
            "intragenic_-rev_comp_": os.path.join(bw_dir, BW_MINUS_RC),
        }
        for p in paths.values():
            if not os.path.exists(p):
                raise FileNotFoundError(f"[FATAL] Missing bigWig: {p}")

        print("[INFO] Mode: STRANDED (4 bigWigs in directory)")
        for k, p in paths.items():
            print(f"  - {k:20s}: {p}")

        handles = {k: bw.open(p) for k, p in paths.items()}
        return "stranded", handles

    if os.path.isfile(bw_dir_or_file) and bw_dir_or_file.endswith(".bw"):
        print("[INFO] Mode: UNSTRANDED (single bigWig file; already averaged)")
        print(f"  - intragenic_unstranded: {bw_dir_or_file}")
        handles = {"intragenic_unstranded": bw.open(bw_dir_or_file)}
        return "unstranded", handles

    raise FileNotFoundError(f"[FATAL] --bw_dir must be a directory or a .bw file: {bw_dir_or_file}")


def close_handles(handles: Dict[str, bw.pyBigWig]):
    for h in handles.values():
        try:
            h.close()
        except Exception:
            pass


def filter_bed_by_intragenic(pairs, args_bw_dir, args_prob_threshold,
                             args_zero_fraction_drop_threshold, args_bed_out):

    intervals = pairs
    if not intervals:
        print("[WARN] No intervals to process. Writing empty output and exiting.")
        os.makedirs(os.path.dirname(os.path.abspath(args_bed_out)), exist_ok=True)
        with open(args_bed_out, "w") as _:
            pass
        return

    mode, handles = open_bw_inputs(args_bw_dir)

    unique_chroms = sorted({iv[0] for iv in intervals})
    ensure_chrom_available(handles, unique_chroms)

    print(f"[INFO] Thresholds: prob_threshold={args_prob_threshold}, "
          f"zero_fraction_drop_threshold={args_zero_fraction_drop_threshold}")
    print(f"[INFO] Processing {len(intervals):,} intervals...")

    kept = []
    n_dropped = 0

    for chrom, start, end, strand, extra in tqdm(intervals, desc="Filtering", unit="interval"):
        if end <= start:
            n_dropped += 1
            continue

        L = end - start

        if mode == "unstranded":
            signal = fetch_values(handles["intragenic_unstranded"], chrom, start, end)
            if len(signal) <= 0:
                n_dropped += 1
                continue
            if len(signal) != L:
                signal = signal[:min(len(signal), L)]
                L = len(signal)

        else:
            v_plus = fetch_values(handles["intragenic_+"], chrom, start, end)
            v_minus = fetch_values(handles["intragenic_-"], chrom, start, end)
            v_plus_rc = fetch_values(handles["intragenic_+rev_comp_"], chrom, start, end)
            v_minus_rc = fetch_values(handles["intragenic_-rev_comp_"], chrom, start, end)

            minL = min(len(v_plus), len(v_minus), len(v_plus_rc), len(v_minus_rc))
            if minL <= 0:
                n_dropped += 1
                continue
            if minL != L:
                v_plus = v_plus[:minL]
                v_minus = v_minus[:minL]
                v_plus_rc = v_plus_rc[:minL]
                v_minus_rc = v_minus_rc[:minL]
                L = minL

            plus_signal = 0.5 * (v_plus + v_minus_rc)
            minus_signal = 0.5 * (v_minus + v_plus_rc)

            if strand == "+":
                signal = plus_signal
            elif strand == "-":
                signal = minus_signal
            else:
                signal = np.maximum(plus_signal, minus_signal)

        binary = (signal > args_prob_threshold).astype(np.uint8)
        zero_fraction = float((binary == 0).mean())

        if zero_fraction > args_zero_fraction_drop_threshold:
            n_dropped += 1
        else:
            kept.append((chrom, start, end, strand, extra))

    os.makedirs(os.path.dirname(os.path.abspath(args_bed_out)), exist_ok=True)
    with open(args_bed_out, "w") as out:
        for chrom, start, end, strand, extra in kept:
            row = [chrom, str(start), str(end)]
            row.append(strand)
            row += extra
            out.write("\t".join(row) + "\n")

    total = len(intervals)
    kept_n = len(kept)
    print(f"[DONE] Kept {kept_n:,} / {total:,} intervals "
          f"({100.0 * kept_n / total:.2f}%). Dropped {n_dropped:,} ({100.0 * n_dropped / total:.2f}%).")
    print(f"[INFO] Intervals left after filtration: {kept_n:,}")

    close_handles(handles)


filter_bed_by_intragenic(pairs, args_bw_dir, args_prob_threshold,
                         args_zero_fraction_drop_threshold, args_bed_out)
