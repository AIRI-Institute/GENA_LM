#!/usr/bin/env python
import ast
import gzip
import json
import random
from pathlib import Path
from typing import Generator, List, Literal, Optional, Tuple

import click
import pandas as pd
from Bio import Seq, SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

# reproducible randomness
random.seed(42)


class Document:
    def __init__(
        self, sentences: Optional[List[str]] = None, metadata: Optional[dict] = None
    ) -> None:
        """
        Data structure to store subsequences that are forming the chromosome

        Each document contains up to `max(sentence_bound) * max(lengths_bound)` of nts
        """
        if sentences is None:
            self.sentences = []
        else:
            self.sentences = sentences
        self.metadata = metadata

    def append(self, sentence: str) -> None:
        self.sentences.append(sentence)

    def to_text(self) -> str:
        return "\n".join(self.sentences)

    def to_jsonl(self) -> None:
        if self.metadata:
            return json.dumps({"text": self.sentences, **self.metadata})
        else:
            return json.dumps({"text": self.sentences})


def generate_documents(
    chr_sequence: Seq,
    n_augmentations: int,    
    sentences_bounds: Tuple[int] = (50, 100),
    lenghts_bounds: Tuple[int] = (500, 1000),
) -> Generator[Document, None, None]:
    """
    From a single chromosome yield a set of documents that cover that chromosome.
    This operation is done ten-fold. 
    """

    C = len(chr_sequence)  # chromosome length

    for _ in range(n_augmentations):
        q = random.randint(0, 5000)  # random start position from the 5' end
        while q < C:
            s = random.randint(*sentences_bounds)  # number of sentences per document
            d = Document()
            for _ in range(s):
                l = random.randint(*lenghts_bounds)  # length of each sentence
                if q+l > C:
                    l = C - q
                d.append(str(chr_sequence[q : q + l]).upper())
                q += l  # update position for the new sentence
                if q>=C: # we reached the end of seq, no more sentences available
                    break
            yield d


def handle_chromosome(
    chr: SeqIO.SeqRecord,
    outdir: Path,
    n_augmentations: int,
    io_mode: Literal["single_txt", "jsonl", "multiple_txt"] = "single_txt",
):
    """
    For a given chromosome make documents and write them to corresponding files
    """

    if io_mode == "single_txt":
        filename = outdir / f"{chr.name}_documents.txt"
        with filename.open(mode="w") as out:
            for document in generate_documents(chr.seq, n_augmentations=n_augmentations):
                out.write(document.to_text())
                out.write("\n")
    elif io_mode == "jsonl":
        filename = outdir / f"{chr.name}_documents.jsonl"
        with open(filename, mode="w") as out:
            for document in generate_documents(chr.seq, n_augmentations=n_augmentations):
                out.write(document.to_jsonl())
                out.write("\n")
    elif io_mode == "multiple_txt":
        for idx, document in enumerate(generate_documents(chr.seq, n_augmentations=n_augmentations)):
            filename = outdir / f"{chr.name}_document_{idx}.txt"
            with filename.open(mode="w") as out:
                out.write(document.to_text())


def read_single_fasta(fna_file: Path, n_augmentations: int, output_dir: Optional[Path] = None,
                      contigs_split_file: Optional[Path] = None,
                      io_mode: Literal["single_txt", "jsonl", "multiple_txt"] = "single_txt",
                      min_len: Optional[int] = 10000,
                      rc: Optional[bool] = False,
                      ):
    if not output_dir:
        output_dir = Path(".")

    if contigs_split_file is not None:
        contigs = pd.read_csv(contigs_split_file, dtype=str)
        if len(contigs) == 0: # no contig split data available
            contigs_split_file = None
        else:
            contigs = contigs.set_index("name")
            contigs["intervals"] = contigs["intervals"].apply(ast.literal_eval)

    if fna_file.endswith(".gz"):
        fasta_open = gzip.open
    else:
        fasta_open = open
    
    with fasta_open(fna_file,"rt") as input_handle:
        for record in tqdm(SeqIO.parse(input_handle, "fasta")):
            if rc:
                record.name  = "rc_" + record.name # modify record name to have different names of rc-files later
            if ("mitochondrion" not in record.description) and len(record.seq) >= min_len:
                if contigs_split_file is not None:
                    chr_contigs = contigs.loc[record.id,"intervals"]
                    for contig in chr_contigs:
                        contig_start = contig[0]
                        contig_end = contig[1]
                        if contig_end-contig_start < min_len:
                            continue
                        contig_name = "_c"+str(contig_start)+"_"+str(contig_end)
                        if rc:
                            seq = record.seq[contig_start:contig_end]
                        else:
                            seq = Seq.reverse_complement(record.seq[contig_start:contig_end])
                        seq_record = SeqRecord(seq = seq,
                                                id = record.id+contig_name,
                                                name = record.name+contig_name,
                                                description = record.description+contig_name)
                        handle_chromosome(seq_record, outdir=output_dir, io_mode=io_mode,
                                          n_augmentations=n_augmentations)
                else:
                    if rc:
                        record.seq = record.seq.reverse_complement()
                    handle_chromosome(record, outdir=output_dir, io_mode=io_mode, n_augmentations=n_augmentations)

# example usage:
# python create_corpus.py --input-file ./ncbi_dataset/data/GCA_009914755.4/GCA_009914755.4_T2T-CHM13v2.0_genomic.fna \
#  --output-dir data/processed/human/ --io-mode jsonl --min-len 10000

@click.command()
@click.option("--input-file", type=str)
@click.option("--contigs-split-file", type=str)
@click.option("--output-dir", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--io-mode", type=click.Choice(["single_txt", "jsonl", "multiple_txt"]), default="single_txt")
@click.option("--min-len", type=click.INT, default=10000, help="Minimum contig length to be included")
@click.option("--n_augmentations", type=click.INT, default=10, help="Number of times each sequence is randomly shifted")
#@click.option("--rc", is_flag=True, show_default=True, default=False, help="Reverse-complement all sequences")

def cli(input_file, contigs_split_file, output_dir, io_mode, min_len, n_augmentations):
    output_dir.mkdir(parents=True)
    assert n_augmentations>0, "The number of augmentations should be > 0"
    read_single_fasta(input_file, contigs_split_file=contigs_split_file, 
                        output_dir=output_dir, io_mode=io_mode,
                        min_len=min_len, rc=True, n_augmentations=n_augmentations)
    read_single_fasta(input_file, contigs_split_file=contigs_split_file, 
                        output_dir=output_dir, io_mode=io_mode,
                        min_len=min_len, rc=False, n_augmentations=n_augmentations)


if __name__ == "__main__":
    cli()
