# import pandas as pd
# import numpy as np
import json
import random
import click
from pathlib import Path
from typing import Generator, List, Literal, Optional, Tuple

from Bio import Seq, SeqIO
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
    sentences_bounds: Tuple[int] = (50, 100),
    lenghts_bounds: Tuple[int] = (500, 1000),
) -> Generator[Document, None, None]:
    """
    From a single chromosome yield a set of documents that cover that chromosome.
    This operation is done ten-fold. 
    """

    C = len(chr_sequence)  # chromosome length

    for _ in range(10):
        q = random.randint(0, 5000)  # random start position from the 5' end
        while q < C:
            s = random.randint(*sentences_bounds)  # number of sentences per document
            d = Document()
            for _ in range(s):
                l = random.randint(*lenghts_bounds)  # length of each sentence
                d.append(str(chr_sequence[q : q + l]).upper())
                q += l  # update position for the new sentence
            yield d


def handle_chromosome(
    chr: SeqIO.SeqRecord,
    outdir: Path,
    io_mode: Literal["single_txt", "jsonl", "multiple_txt"] = "single_txt",
):
    """
    For a given chromosome make documents and write them to corresponding files
    """

    if io_mode == "single_txt":
        filename = outdir / f"{chr.name}_documents.txt"
        with filename.open(mode="w") as out:
            for document in tqdm(generate_documents(chr.seq), desc=chr.description):
                out.write(document.to_text())
                out.write("\n")
    elif io_mode == "jsonl":
        filename = outdir / f"{chr.name}_documents.jsonl"
        with filename.open(mode="w") as out:
            for document in tqdm(generate_documents(chr.seq), desc=chr.description):
                out.write(document.to_jsonl())
                out.write("\n")
    elif io_mode == "multiple_txt":
        for idx, document in enumerate(generate_documents(chr.seq)):
            filename = outdir / f"{chr.name}_document_{idx}.txt"
            with filename.open(mode="w") as out:
                out.write(document.to_text())


def read_single_fasta(fna_file: Path, output_dir: Optional[Path] = None,
                      io_mode: Literal["single_txt", "jsonl", "multiple_txt"] = "single_txt"):
    if not output_dir:
        output_dir = Path(".")

    with open(fna_file) as input_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            if "mitochondrion" not in record.description:
                handle_chromosome(record, outdir=output_dir, io_mode=io_mode)

# example usage:
# python create_corpus.py --input_file ./ncbi_dataset/data/GCA_009914755.4/GCA_009914755.4_T2T-CHM13v2.0_genomic.fna \
#  --output_dir data/processed/human/ --io_mode jsonl

@click.command()
@click.option("--input_file", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--output_dir", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--io_mode", type=click.Choice(["single_txt", "jsonl", "multiple_txt"]), default="single_txt")
def cli(input_file, output_dir, io_mode):
    output_dir.mkdir(parents=True)
    read_single_fasta(input_file, output_dir=output_dir, io_mode=io_mode)


if __name__ == "__main__":
    cli()