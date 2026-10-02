"""Smoke test for ExpressionDatasetCNN at the Flashzoi geometry, on real data.

Checks the item contract and, most importantly, that the label bins land on the
*centre crop* of the window -- the one thing that would otherwise fail silently.

    GENALM_HOME=... PYTHONPATH=. python .../smoke_dataset_flashzoi.py
"""

import logging
import os
import sys

import numpy as np
import torch

from downstream_tasks.expression_prediction.expression_dataset_cnn import ExpressionDatasetCNN
from downstream_tasks.expression_prediction.expression_dataset_final import logtransform

HOME = os.environ["GENALM_HOME"]
EP = f"{HOME}/GENA_LM/downstream_tasks/expression_prediction"

WINDOW = 32768
BIN_SIZE = 32
OUT_BINS = 1022
N_KEYS = 14


def main():
    ds = ExpressionDatasetCNN(
        forward_intervals_path=f"{EP}/intervals/human.valid.forward.csv",
        reverse_intervals_path=f"{EP}/intervals/human.valid.reverse.csv",
        targets_path=f"{EP}/datasets/data/file_mappings/Expression_dataset_v1_csv_file_mappings_qnorm.csv",
        genome=f"{EP}/datasets/data/genomes/hg38/hg38.fa",
        dna_window_len=WINDOW,
        cnn_total_stride=BIN_SIZE,
        out_bins=OUT_BINS,
        n_keys=N_KEYS,
        tpm="csv",
        transform_targets_tpm=logtransform(pseudocount=1),
        loglevel=logging.INFO,
    )
    print("[ds]", ds.describe())

    assert ds.n_bins_full == WINDOW // BIN_SIZE == 1024
    assert ds.n_bins == OUT_BINS
    assert ds.crop_offset == 32, ds.crop_offset
    assert ds.seq_len == 1024

    failures = []
    for idx in (0, 1, len(ds) // 2, len(ds) - 1):
        item = ds[idx]
        assert item["dna_codes"].shape == (WINDOW,), item["dna_codes"].shape
        assert item["dna_codes"].dtype == torch.uint8
        assert item["attention_mask"].shape == (1024,)
        assert item["labels"].shape == (N_KEYS, 1024, 1)
        assert item["labels_mask"].shape == (N_KEYS, 1024, 1)
        # tpm-only: nothing but position 0 is a target
        assert not item["labels_mask"][:, 1:, :].any(), "per-bin targets in a tpm-only dataset"
        assert item["labels_mask"][:, 0, 0].any(), "no TPM target at all"
        assert len(item["desc_input_ids"]) == N_KEYS

        # Geometry: the label bins must cover the centre 1022*32 bp of the window,
        # 32 bp in from each edge, and stay centred on the TSS.
        row = ds.genes.iloc[ds.valid_indices[idx // ds.n_cell_chunks]]
        start, end = ds._window_for_gene(row)
        starts, ends = ds._bin_coords(start, end, row["strand"])
        assert len(starts) == OUT_BINS
        lo, hi = int(starts.min()), int(ends.max())
        if (lo, hi) != (start + 32, end - 32):
            failures.append(f"idx={idx}: bins span {lo}-{hi}, expected {start + 32}-{end - 32}")
        if hi - lo != OUT_BINS * BIN_SIZE:
            failures.append(f"idx={idx}: span {hi - lo} != {OUT_BINS * BIN_SIZE}")
        # bins are contiguous and ordered along transcription
        order = np.argsort(starts) if row["strand"] == "+" else np.argsort(-starts)
        if not np.array_equal(order, np.arange(OUT_BINS)):
            failures.append(f"idx={idx}: bins not ordered along transcription")

        tss = int(row["TSS"])
        if not (lo <= tss < hi):
            failures.append(f"idx={idx}: TSS {tss} outside the label region {lo}-{hi}")

    if failures:
        print("[ds] FAIL")
        for f in failures:
            print("   -", f)
        return 1

    print(f"[ds] OK {len(ds):,} items, {len(ds.valid_indices):,} genes x {N_KEYS} tracks")
    print(f"[ds] window {WINDOW} bp -> labels on the centre {OUT_BINS * BIN_SIZE} bp, "
          f"{ds.crop_offset} bp cropped per edge")
    return 0


if __name__ == "__main__":
    sys.exit(main())
