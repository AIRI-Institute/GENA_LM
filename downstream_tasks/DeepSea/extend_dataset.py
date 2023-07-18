import argparse
import gzip
import os
import subprocess

import pysam
from tqdm import tqdm
from Bio.Seq import Seq 

def get_region(fasta, chrom, mid, size, strand):
    start = mid - size // 2
    start = max(start, 0)
    end = start + size
    seq = fasta.fetch(chrom, start, end).upper()
    if strand == "-":
        seq = Seq(seq).reverse_complement()

    if len(seq) < size:
        return None
    elif len(seq) == size:
        return seq
    else:
        raise ValueError("Unexpected fasta segment at posistion " + str(chrom) + ":" + str(mid))


parser = argparse.ArgumentParser(
    description="Extend sequence length for deepSea dataset\nexample usage:\npython extend_dataset.py --path ~/DNALM/downstream_tasks/DeepSea_chromatin_profile/ --fasta /mnt/datasets/DeepCT/male.hg19.fasta --seqlen 8000",
)
parser.add_argument(
    "--path",
    required=True,
    help="path to the folder containing .csv.gz original dataset files (train, test, and valid)",
)
parser.add_argument("--seqlen", required=True, type=int, help="desired sequence length")
parser.add_argument(
    "--fasta",
    required=True,
    help="path to fasta & index file \n (note - fasta should be bwa-indexed, i.e. run bwa index /path/to/fasta first)",
)

args = parser.parse_args()

path = args.path
new_seq_len = args.seqlen
fasta = pysam.FastaFile(args.fasta)

for fname in ["test", "valid", "train"]:
    print(f"Processing dataset split {fname}")
    fastmap_file_path = os.path.join(path, fname + ".fastmap")
    if not os.path.exists(fastmap_file_path):
        cmd = (
            "zcat "
            + os.path.join(path, fname + ".csv.gz")
            + " | "
            + 'awk -F "," \'BEGIN{i=0}{print ">" i "\\n" $1; i++}\''
            + " | bwa fastmap -l 1000 "
            + args.fasta
            + " - 2> /dev/null 1>"
            + fastmap_file_path
        )
        print(f"Running cmd \n{cmd}")
        subprocess.getoutput(cmd)
    else:
        print("Using existing fastmap file")

    assert os.path.exists(fastmap_file_path)

    unmapped_count = 0
    total_count = 0

    with gzip.open(os.path.join(path, fname + ".csv.gz")) as target_in, open(
        fastmap_file_path
    ) as fastmap_in, gzip.open(os.path.join(path, fname + "." + str(new_seq_len) + ".csv.gz"), "w") as out:
        for ind, line in tqdm(enumerate(target_in)):
            line = line.decode()
            split = line.find(",")
            seq = line[:split]
            targets = line[split:]  # includes starting coma and newline char

            fmap_header = fastmap_in.readline().strip().split()
            fmap_aln = fastmap_in.readline().strip().split()
            if fmap_aln[0] == "//":  # no match found for this seq
                # assert set(list(seq)) == "N", "No matches found for seq \n"+seq+"\nid="+str(ind)
                # new_seq = "".join(["N"]*new_seq_len)
                new_seq = seq
                unmapped_count += 1
            else:
                fmap_footer = fastmap_in.readline().strip()

                assert fmap_footer == "//", fmap_footer

                assert len(fmap_header) == 3, fmap_header
                assert fmap_header[0] == "SQ", fmap_header
                assert fmap_header[1] == str(ind), fmap_header
                assert fmap_header[2] == "1000", fmap_header

                # EM	0	1000	1	chr8:+20601
                assert fmap_aln[0] == "EM", fmap_aln
                if len(fmap_aln) != 5 or fmap_aln[2] != "1000" or fmap_aln[1] != "0" or fmap_aln[3] != "1":
                    new_seq = seq
                    unmapped_count += 1
                else:
                    chrom, pos = fmap_aln[4].split(":")
                    assert pos[0] in ["+","-"], str(fmap_aln)
                    strand = pos[0]
                    pos = abs(int(pos))
                    new_seq = get_region(fasta, chrom, pos + 500, new_seq_len, strand)

            assert new_seq is not None, str(ind)
            out.write((new_seq + targets).encode())
            total_count += 1
    print("Unmapped: ", unmapped_count, " out of total = ", total_count)