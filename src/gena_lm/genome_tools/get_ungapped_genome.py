import os
import pysam
from tqdm import tqdm

def split_fasta_by_gaps(input_fasta: str, output_fasta: str) -> None:
    """
    Split a FASTA file into gapless segments (regions without 'N') and write them to a new FASTA file.

    Each segment is defined between consecutive 'N' bases (uppercase or lowercase) in the reference sequence.
    The output FASTA records will have headers of the form:
        >chromosome:start-end
    using 0-based coordinates.

    Parameters
    ----------
    input_fasta : str
        Path to the input FASTA file (must be indexed if using pysam.FastaFile).
    output_fasta : str
        Path for the output FASTA file to write the segments.

    Raises
    ------
    AssertionError
        If the input file does not exist or sequences cannot be fetched.
    """
    # Validate input path
    assert os.path.isfile(input_fasta), f"Input FASTA not found: {input_fasta}"

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_fasta))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Open input FASTA via pysam
    fasta_in = pysam.FastaFile(input_fasta)

    # Open output FASTA for writing
    with open(output_fasta, 'w') as out_fh:
        # Iterate through each reference (chromosome/contig)
        for chrom in fasta_in.references:
            print(f"Processing chromosome: {chrom}")
            # Fetch the entire sequence and normalize to uppercase
            seq = fasta_in.fetch(chrom).upper()
            assert isinstance(seq, str), f"Sequence fetch for {chrom} failed"

            # Track the start index of the current non-gap segment
            seg_start = 0
            length = len(seq)

            # Iterate over sequence positions
            for idx, base in tqdm(enumerate(seq), total=length):
                if base == 'N':
                    # If there's a non-empty segment before this N, write it
                    if seg_start < idx:
                        seg_seq = seq[seg_start:idx]
                        # Use 0-based coordinates
                        header = f">{chrom}:{seg_start}-{idx}"
                        out_fh.write(f"{header}\n{seg_seq}\n")
                    # Move start to the position after the N
                    seg_start = idx + 1

            # Handle any segment after the last N to the end
            if seg_start < length:
                seg_seq = seq[seg_start:length]
                header = f">{chrom}:{seg_start}-{length}"
                out_fh.write(f"{header}\n{seg_seq}\n")

    # Close input
    fasta_in.close()


# Example usage
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Split a FASTA into gapless segments.')
    parser.add_argument('input_fasta', help='Input FASTA file path')
    parser.add_argument('output_fasta', help='Output FASTA file path')
    args = parser.parse_args()

    split_fasta_by_gaps(args.input_fasta, args.output_fasta)
