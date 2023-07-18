#!/usr/bin/env python
import click
import json
import random

from pathlib import Path
from typing import Generator, List, Literal, Optional, Tuple

from Bio import Seq, SeqIO
from cyvcf2 import VCF, Variant
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

    def set_metadata(self, metadata: dict) -> None:
        self.metadata = metadata

    def to_text(self) -> str:
        return "\n".join(self.sentences)

    def to_jsonl(self) -> str:
        if self.metadata:
            return json.dumps({"text": self.sentences, **self.metadata})
        else:
            return json.dumps({"text": self.sentences})


def generate_consensus_documents(
    chr_sequence: SeqIO.SeqRecord,
    vcf_file: VCF,
    sample_list: List[str],
    sentences_bounds: Tuple[int, int] = (50, 100),
    lengths_bounds: Tuple[int, int] = (500, 1000),
) -> Generator[Document, None, None]:
    """
    Yield documents covering the chromosome from a reference
    replacing its positions with alleles from a VCF file for each sample in it.
    """

    def choose_alleles(
        genotype: List,
        ref_allele: str,
        alt_alleles: List[str],
        AF: float,
    ) -> Tuple[str, str, int, int]:
        """
        Helper function to return two allele letters.
        cyvcf2 returns genotype as [allele1:int, allele2:int, phase_status]
        """
        if genotype[0] == -1:
            # not genotyped, will use reference
            return (ref_allele, ref_allele, 0, 0)
        else:
            alleles = [ref_allele] + alt_alleles
            
            is_common = 1
            if AF < 0.1:
                is_common = 0

            return (alleles[genotype[0]], alleles[genotype[1]], 1, is_common)

    def make_sample_use_mask(
        samples: List[str],
        variants: List[Variant]
    ) -> List[bool]:
        """
        Helper function to understand if we should skip any of samples.
        Returns a bitmask over all samples: False = skip, True = use
        """
        sample_use_mask = [False for _ in samples]
        for sample_id in range(len(samples)):
            # check every sample if it has genotyped variants in this region
            for variant in variants:
                if variant.genotypes[sample_id][0] not in [0, -1]:
                    # sample actually is genotyped and not a reference
                    sample_use_mask[sample_id] = True
                    break
        return sample_use_mask

    def prepare_documents(
        sequence: str,
        snt_bounds: List[int],
        metadata: dict
    ) -> Tuple[Document, Document]:
        """
        Helper function for a final compilation of documents.
        Given a ready full sequence and sentence bounds,
        split it into sentences and repeat it for the revcomp copy
        """
        d = Document(metadata=metadata)
        d_rc = Document(metadata=metadata)

        snt_pos = 0
        for snt_len in snt_bounds:
            snt = sequence[snt_pos:snt_pos+snt_len]
            d.append(snt)
            snt_pos += snt_len

        snt_pos = 0
        sequence_rc = Seq.reverse_complement(sequence)
        for snt_len in reversed(snt_bounds):
            snt = sequence_rc[snt_pos:snt_pos+snt_len]
            d_rc.append(snt)
            snt_pos += snt_len
        return (d, d_rc)

    C = len(chr_sequence)

    for _ in range(1):
        doc_start = random.randint(0, 5000) 
        while doc_start + lengths_bounds[1] < C:
            # prepare a document
            current_pos = doc_start
            
            snt_len = random.randint(*lengths_bounds)
            snt_left = random.randint(*sentences_bounds)
            snt_bounds = []
            while (current_pos + snt_len < C) and (snt_left > 0):
                # prepare a sentence in this document
                snt_left -= 1
                current_pos += snt_len
                snt_bounds.append(snt_len)
                snt_len = random.randint(*lengths_bounds)
            
            doc_end = current_pos
            ref_part = str(chr_sequence.seq[doc_start:doc_end].upper())
            ref_len = len(ref_part)
            NF = ref_part.count("N") / ref_len
            if NF > 0.5:
                doc_start = doc_end
                continue

            region = f"{chr_sequence.id}:{doc_start}-{doc_end}"
            variants = list(vcf_file(region))
            if not variants:
                doc_start = doc_end
                continue

            sample_use_mask = make_sample_use_mask(vcf_file.samples, variants)
            for sample_id in range(len(vcf_file.samples)):
                if not sample_use_mask[sample_id]:
                    continue

                sample_cons1 = ""
                sample_cons2 = ""
                total_VCS = 0
                total_CCS = 0
                total_CC = 0
                total_RC = 0
                var_pos = 0
                for v in variants:
                    # -1: v.POS is VCF (1-based) and doc_start is python (0-based)
                    local_pos = v.POS - doc_start - 1
                    if local_pos > var_pos:
                        # parts prior to variant
                        sample_cons1 += ref_part[var_pos:local_pos]
                        sample_cons2 += ref_part[var_pos:local_pos]
                    
#                   AF = v.INFO.get("gnomad_AF")
#                   if not AF:
#                       AF = v.INFO.get("AF")
#                   if not AF:
#                       AF = v.aaf
# requesting AF slows it down 3-10x
                    AF = 0.5

                    # variant alleles
                    (allele1, allele2, variant_count, common_count) = choose_alleles(
                        v.genotypes[sample_id],
                        v.REF, v.ALT, AF
                    )
                    sample_cons1 += allele1
                    sample_cons2 += allele2

                    total_VCS += variant_count
                    total_CCS += common_count

#                   if AF < 0.01:
#                       total_RC += 1
#                   if AF > 0.1:
#                       total_CC += 1

                    # +1: we already selected var_pos+0 for the variant itself
                    # -1: if REF is long (in case of deletion), 
                    # skip letters after the first one
                    # as choose_alleles() already added right amount of REF
                    # if sample was REF
                    var_pos = local_pos + 1 + len(v.REF) - 1

                if (total_CCS / ref_len < 0.01):
                    # skip region with low common
                    continue
                
                metadata = {
                    "VCS": total_VCS,                       # applied variant count
#                   "CCS": total_CCS,                       # applied common >10% count
#                   "RC": total_RC,                         # pop rare <1% count
#                   "CC": total_CC,                         # pop common count
                    "VF": f"{total_VCS/len(ref_part):.2f}", # VCS normalized by doc len
                    "sample": vcf_file.samples[sample_id],
                    "len": len(ref_part),
                    "snt": len(snt_bounds)
                }

                cons_unequal = True
                if sample_cons1 == sample_cons2:
                    cons_unequal = False

                # parts after the last variant
                sample_cons1 += ref_part[var_pos:]

                (d1, d1_rc) = prepare_documents(sample_cons1, snt_bounds, metadata)
                yield d1
                yield d1_rc

                if cons_unequal:
                    sample_cons2 += ref_part[var_pos:]
                    (d2, d2_rc) = prepare_documents(sample_cons2, snt_bounds, metadata)
                    yield d2
                    yield d2_rc

            doc_start = doc_end


def handle_chromosome_and_variants(
    chr: SeqIO.SeqRecord,
    vcf_file: VCF,
    sample_list: List[str],
    outdir: Path,
    io_mode: Literal["single_txt", "jsonl", "multiple_txt"] = "single_txt",
):
    """
    For a given chromosome and variants make documents and write them to corresponding files
    """
    samples = ",".join(sample_list)

    if io_mode == "single_txt":
        filename = outdir / f"{chr.id}.{samples}.documents.txt"
        with filename.open(mode="w") as out:
            for document in tqdm(generate_consensus_documents(chr, vcf_file, sample_list), desc=chr.description):
                out.write(document.to_text())
                out.write("\n")
    elif io_mode == "jsonl":
        filename = outdir / f"{chr.id}.{samples}.documents.jsonl"
        with filename.open(mode="w") as out:
            for document in tqdm(generate_consensus_documents(chr, vcf_file, sample_list), desc=chr.description):
                out.write(document.to_jsonl())
                out.write("\n")
    elif io_mode == "multiple_txt":
        for idx, document in enumerate(generate_consensus_documents(chr, vcf_file, sample_list)):
            filename = outdir / f"{chr.id}.{samples}.document_{idx}.txt"
            with filename.open(mode="w") as out:
                out.write(document.to_text())


def read_fasta_and_vcf(
    fasta_file: Path,
    vcf_file: Path,
    samples: str,
    output_dir: Optional[Path] = None,
    io_mode: Literal["single_txt", "jsonl", "multiple_txt"] = "single_txt"
) -> None:
    if not output_dir:
        output_dir = Path(".")

    sample_list = samples.split(",")
    vcf = VCF(vcf_file, samples=sample_list, threads=2)
    
    with open(fasta_file) as input_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            if record.id in ("chrM"):
                continue
            try:
                x = next(vcf(record.id))
            except StopIteration:
                continue
            handle_chromosome_and_variants(
                record,
                vcf,
                sample_list,
                outdir=output_dir, io_mode=io_mode
            )


# example usage:
# python create_corpus.py --input-file /path/to/hg38.fa \
#  --vcf /path/to/gnomad.vcf \
#  --samples NA12877,NA12878 \
#  --output_dir data/processed/alleles/ --io_mode jsonl

@click.command()
@click.option("--reference", required=True,
              help="Path to the genome reference fasta",
              type=click.Path(path_type=Path, dir_okay=False))
@click.option("--vcf", required=True,
              help="Path to the indexed vcf",
              type=click.Path(path_type=Path, dir_okay=False))
@click.option("--samples", required=False,
              help="""Comma-separated list of samples from the vcf to use.
                   By default all will be used.""")
@click.option("--output-dir", required=False,
              type=click.Path(path_type=Path, dir_okay=True))
@click.option("--io-mode", 
              type=click.Choice(["single_txt", "jsonl", "multiple_txt"]),
              default="jsonl")
def cli(reference, vcf, samples, output_dir, io_mode):
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    read_fasta_and_vcf(reference, vcf, samples, output_dir=output_dir, io_mode=io_mode)


if __name__ == "__main__":
    cli()
