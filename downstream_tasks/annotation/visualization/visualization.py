import os
import numpy as np
import pandas as pd
import pyBigWig

def write_array_dense_bw(arr, chrom, start, fai_path, out_bw):
    """
    Write a numpy array to a BigWig file in dense mode (one value per base).

    Parameters
    ----------
    arr : np.array
        Array of values to be written (1D).
    chrom : str
        Chromosome name (must be present in the FAI file).
    start : int
        Start position on the chromosome (0-based).
    fai_path : str
        Path to the FASTA index (.fai) file.
    out_bw : str
        Output BigWig file path.
    """
    df = pd.read_csv(
            fai_path,
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["chrom", "length"]
        )
    chrom_sizes = dict(zip(df["chrom"], df["length"]))

    if chrom not in chrom_sizes:
        raise ValueError(f"Chromosome {chrom} not found in FAI")
    if start < 0 or start + len(arr) > chrom_sizes[chrom]:
        raise ValueError("Interval goes beyond chromosome boundaries")

    if os.path.exists(out_bw):
        os.remove(out_bw)

    bw = pyBigWig.open(out_bw, "w")
    bw.addHeader(list(chrom_sizes.items()))

    bw.addEntries(chrom, start, values=list(arr.astype(float)), span=1, step=1)

    bw.close()

#Example
pred = np.array([5, 2, 3, 5, 5, 2, 2, 7, 7, 7], dtype=float)
write_array_dense_bw(pred, chrom="chr20", start=0,
                     fai_path="GRCh38.primary_assembly.genome.fa.fai", out_bw="signal_dense.bw")