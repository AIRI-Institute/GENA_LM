"""Is our one-hot the one Borzoi was trained with?

Nothing raises if the channel order is wrong -- the model just gets garbage -- so
this is checked against borzoi-pytorch's own fixture ``wt_seq.npy``, the canonical
one-hot the repo uses to reproduce the TensorFlow predictions exactly.

Three independent checks:

1. Decode the fixture under our A,C,G,T channel order and look for CpG depletion.
   Vertebrate DNA has ~5x fewer CG than GC dinucleotides. Complementing a strand
   (C<->G) turns every CG into a GC, so a wrong order flips this ratio. This is
   decisive on its own.
2. Find the decoded sequence verbatim in hg38. A complemented-but-not-reversed
   sequence does not occur in the genome, so an exact hit settles it.
3. Round-trip: run the decoded string back through our own
   encode_nucleotides + build_onehot_lookup and require it to reproduce the
   fixture bit for bit. That is the pipeline the dataset and the encoder use.

    GENALM_HOME=... PYTHONPATH=. python .../verify_dna_encoding.py wt_seq.npy
"""

import sys

import numpy as np
import torch

from downstream_tasks.expression_prediction.alphagenome_cnn import (
    build_onehot_lookup,
    encode_nucleotides,
    reverse_complement_codes,
)

HG38 = None  # set in main


def decode(onehot, order):
    idx = onehot.argmax(axis=1)
    return "".join(order[i] for i in idx)


def dinuc(seq, pair):
    n, start = 0, 0
    while True:
        i = seq.find(pair, start)
        if i < 0:
            return n
        n += 1
        start = i + 1


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "wt_seq.npy"
    onehot = np.load(path)
    print(f"[fixture] {path} shape={onehot.shape} dtype={onehot.dtype}")
    assert onehot.shape[1] == 4

    # --- 1. CpG depletion under each candidate order -----------------------
    print("\n[1] CpG vs GpC (real vertebrate DNA: CG << GC)")
    verdict = {}
    for order in ("ACGT", "TGCA", "AGCT", "ACTG"):
        seq = decode(onehot, order)
        cg, gc = dinuc(seq, "CG"), dinuc(seq, "GC")
        ratio = cg / max(gc, 1)
        verdict[order] = ratio
        flag = "<- looks like real DNA" if ratio < 0.5 else ""
        print(f"    order={order}: CG={cg:>7,} GC={gc:>7,} CG/GC={ratio:.3f} {flag}")

    best = min(verdict, key=verdict.get)
    print(f"    => most depleted in CpG: {best}")
    assert best == "ACGT", f"our LUT assumes ACGT but the fixture looks like {best}"

    seq = decode(onehot, "ACGT")

    # --- 2. exact hg38 match ------------------------------------------------
    print("\n[2] verbatim match in hg38")
    probe = seq[len(seq) // 2 : len(seq) // 2 + 120]
    print(f"    probe (120bp from the centre): {probe[:60]}...")
    from pysam import FastaFile

    fa = FastaFile(HG38)
    hit = None
    for chrom in [c for c in fa.references if "_" not in c][:25]:
        text = fa.fetch(chrom).upper()
        pos = text.find(probe)
        if pos >= 0:
            hit = (chrom, pos)
            break
        pos = text.find(str.translate(probe, str.maketrans("ACGT", "TGCA"))[::-1])
        if pos >= 0:
            hit = (chrom, pos, "reverse strand")
            break
    if hit:
        print(f"    FOUND {hit}")
    else:
        print("    not found (fixture may be from another assembly); check 1 still stands")

    # --- 3. round-trip through our own pipeline -----------------------------
    print("\n[3] round-trip: string -> encode_nucleotides -> build_onehot_lookup")
    codes = encode_nucleotides(seq)
    assert codes.dtype == np.uint8 and codes.shape == (len(seq),)
    lut = build_onehot_lookup()
    ours = torch.nn.functional.embedding(torch.as_tensor(codes).long(), lut).numpy()
    same = np.array_equal(ours, onehot)
    print(f"    reproduces the fixture exactly: {same}")
    assert same, "our one-hot differs from borzoi's canonical one-hot"

    # channel-first, as conv_dna wants
    chan_first = torch.as_tensor(ours).transpose(0, 1)
    assert chan_first.shape == (4, len(seq))
    print(f"    channel-first shape for conv_dna: {tuple(chan_first.shape)}")

    # --- 4. N handling ------------------------------------------------------
    print("\n[4] N / ambiguity codes")
    n_codes = encode_nucleotides("ACGTNRYacgt")
    n_onehot = torch.nn.functional.embedding(torch.as_tensor(n_codes).long(), lut).numpy()
    print(f"    'ACGTNRYacgt' -> codes {n_codes.tolist()}")
    print(f"    N row: {n_onehot[4].tolist()}, R row: {n_onehot[5].tolist()}")
    assert n_onehot[4].sum() == 0 and n_onehot[5].sum() == 0, "N must be all-zeros"
    assert n_onehot[7:].sum() == 0, "lowercase acgt should NOT be silently encoded"
    print("    OK N and IUPAC codes -> all-zero rows (Borzoi's convention)")
    print("    NOTE lowercase is NOT handled -- the dataset calls .upper() before encoding")

    # --- 5. reverse complement ---------------------------------------------
    print("\n[5] reverse complement in code space (dataset does this for '-' genes)")
    rc = reverse_complement_codes(encode_nucleotides("AACGTTN"))
    rc_str = "".join("ACGTN"[c] for c in rc)
    print(f"    revcomp('AACGTTN') = {rc_str!r}")
    assert rc_str == "NAACGTT", rc_str
    # upstream borzoi spells revcomp as x.flip(dims=(1,2)): reversing the channel
    # axis must equal complementing, which only holds for ACGT ordering.
    x = torch.as_tensor(ours[:16]).transpose(0, 1)[None]          # (1, 4, 16)
    flipped = x.flip(dims=(1, 2))[0].transpose(0, 1).numpy()
    ours_rc = torch.nn.functional.embedding(
        torch.as_tensor(reverse_complement_codes(codes[:16]).copy()).long(), lut
    ).numpy()
    assert np.array_equal(flipped, ours_rc), (
        "our code-space revcomp disagrees with borzoi's flip(dims=(1,2))"
    )
    print("    OK matches borzoi's own flip(dims=(1,2)) revcomp")

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    import os

    HOME = os.environ["GENALM_HOME"]
    HG38 = f"{HOME}/GENA_LM/downstream_tasks/expression_prediction/datasets/data/genomes/hg38/hg38.fa"
    sys.exit(main())
