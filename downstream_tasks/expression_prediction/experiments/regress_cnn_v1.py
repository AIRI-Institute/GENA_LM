"""Did the out_bins patch change ExpressionDatasetCNN for cnn_v1?

Loads the pre-patch module from the .bak file and the patched one side by side,
builds both at cnn_v1's geometry (8176 bp / stride 8, no out_bins) and requires
them to agree on cache paths, geometry and the actual tensors.

Cache paths matter as much as the tensors: a changed hash silently orphans every
precomputed cache cnn_v1 already has.

    GENALM_HOME=... PYTHONPATH=. python .../regress_cnn_v1.py
"""

import glob
import importlib.machinery
import importlib.util
import logging
import os
import sys

import numpy as np
import torch

HOME = os.environ["GENALM_HOME"]
EP = f"{HOME}/GENA_LM/downstream_tasks/expression_prediction"
# The pre-patch copy. Its date differs per machine, so it is discovered rather
# than hardcoded; override with BAK=... for an explicit file.
BAK = os.environ.get("BAK") or sorted(
    glob.glob(f"{EP}/expression_dataset_cnn.py.bak.*")
)[-1]

# cnn_v1.yaml: DNA_WINDOW_LEN 8176, CNN_TOTAL_STRIDE 8, n_keys 14
WINDOW, STRIDE, N_KEYS = 8176, 8, 14


def load_old():
    # .bak.<date> is not a recognised source suffix, so the loader is explicit.
    loader = importlib.machinery.SourceFileLoader("expression_dataset_cnn_old", BAK)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.ExpressionDatasetCNN


def build(cls):
    from downstream_tasks.expression_prediction.expression_dataset_final import logtransform

    return cls(
        forward_intervals_path=f"{EP}/intervals/human.valid.forward.csv",
        reverse_intervals_path=f"{EP}/intervals/human.valid.reverse.csv",
        targets_path=f"{EP}/datasets/data/file_mappings/Expression_dataset_v1_csv_file_mappings_qnorm.csv",
        genome=f"{EP}/datasets/data/genomes/hg38/hg38.fa",
        dna_window_len=WINDOW,
        cnn_total_stride=STRIDE,
        n_keys=N_KEYS,
        tpm="csv",
        transform_targets_tpm=logtransform(pseudocount=1),
        loglevel=logging.WARNING,
    )


def main():
    from downstream_tasks.expression_prediction.expression_dataset_cnn import ExpressionDatasetCNN

    old = build(load_old())
    new = build(ExpressionDatasetCNN)
    bad = []

    print("[paths] cache hashes -- a change here orphans cnn_v1's existing caches")
    for name in ("get_hash_path", "get_signals_hash_path", "get_tpm_hash_path"):
        a, b = getattr(old, name)(), getattr(new, name)()
        ok = a == b
        print(f"    {name:<24} {'SAME' if ok else 'DIFFERENT'}  {os.path.basename(b)}")
        if not ok:
            bad.append(f"{name}: {a} != {b}")

    print("\n[geometry]")
    for attr in ("n_bins", "seq_len", "dna_window_len", "cnn_total_stride", "window_offset"):
        a, b = getattr(old, attr), getattr(new, attr)
        print(f"    {attr:<24} old={a} new={b}")
        if a != b:
            bad.append(f"{attr}: {a} != {b}")
    print(f"    crop_offset (new only)   {new.crop_offset}  <- must be 0 for cnn_v1")
    if new.crop_offset != 0:
        bad.append(f"crop_offset is {new.crop_offset}, expected 0")
    if len(old) != len(new):
        bad.append(f"len: {len(old)} != {len(new)}")

    print("\n[bin coords] over 200 genes")
    for gi in range(0, min(200, len(old.genes))):
        row = old.genes.iloc[gi]
        wa, wb = old._window_for_gene(row), new._window_for_gene(row)
        if wa != wb:
            bad.append(f"gene {gi}: window {wa} != {wb}")
            continue
        sa, ea = old._bin_coords(*wa, row["strand"])
        sb, eb = new._bin_coords(*wb, row["strand"])
        if not (np.array_equal(sa, sb) and np.array_equal(ea, eb)):
            bad.append(f"gene {gi}: bin coords differ")
    print(f"    {'OK' if not bad else 'MISMATCH'}")

    print("\n[items] full tensor comparison over 12 items")
    idxs = np.linspace(0, len(old) - 1, 12).astype(int)
    for idx in idxs:
        ia, ib = old[int(idx)], new[int(idx)]
        if ia.keys() != ib.keys():
            bad.append(f"item {idx}: key sets differ")
            continue
        for k, va in ia.items():
            vb = ib[k]
            if isinstance(va, torch.Tensor):
                if not torch.equal(va, vb):
                    bad.append(f"item {idx}: tensor {k} differs")
            elif isinstance(va, list) and va and isinstance(va[0], torch.Tensor):
                if not all(torch.equal(x, y) for x, y in zip(va, vb)):
                    bad.append(f"item {idx}: tensor list {k} differs")
            elif va != vb:
                bad.append(f"item {idx}: {k}: {va!r} != {vb!r}")
    print(f"    {'OK' if not bad else 'MISMATCH'}")

    if bad:
        print(f"\nREGRESSION: {len(bad)} difference(s)")
        for b in bad[:20]:
            print("   -", b)
        return 1
    print("\nOK the patch is a no-op for cnn_v1: same caches, same geometry, same tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
