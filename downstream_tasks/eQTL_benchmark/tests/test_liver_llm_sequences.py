import csv
import gzip
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from liver_llm_sequences import (
    build_alt_variant_region,
    build_liver_llm_sequence_records,
    centered_start0,
    fetch_centered_sequence,
    parse_variant_token,
    to_long_records,
    write_records,
)


class FixedGenome:
    def __init__(self, sequence: str):
        self.sequence = sequence.upper()

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        bases = []
        for pos0 in range(start0, end0):
            if 0 <= pos0 < len(self.sequence):
                bases.append(self.sequence[pos0])
            else:
                bases.append("N")
        return "".join(bases)


def test_parse_variant_token_normalizes_chromosome():
    variant = parse_variant_token("7-150499647-G-A")

    assert variant.chrom == "chr7"
    assert variant.pos == 150499647
    assert variant.ref == "G"
    assert variant.alt == "A"
    assert variant.variant_id == "chr7-150499647-G-A"


def test_centered_even_window_places_position_at_half_offset():
    assert centered_start0(101, 10) == 95

    genome = FixedGenome("N" * 100 + "ACGTAA")
    seq, start0 = fetch_centered_sequence(genome, "chr1", 101, 10)

    assert start0 == 95
    assert len(seq) == 10
    assert seq[5] == "A"


def test_build_alt_variant_region_handles_snv_and_indels_at_fixed_length():
    assert build_alt_variant_region("CCCCACCCCC", 4, "A", "T", 10) == "CCCCTCCCCC"

    insertion = build_alt_variant_region("CCCCACCCCC", 4, "A", "ATG", 10)
    deletion = build_alt_variant_region("CCCCATCCCCGG", 4, "AT", "A", 10)

    assert insertion == "CCCCATGCCC"
    assert deletion == "CCCCACCCCG"
    assert len(insertion) == 10
    assert len(deletion) == 10


def test_ref_mismatch_raises_by_default():
    with unittest.TestCase().assertRaisesRegex(ValueError, "FASTA REF allele mismatch"):
        build_alt_variant_region("CCCCGCCCCC", 4, "A", "T", 10)


def test_build_records_expands_variants_by_zero_expression_genes(tmp_path):
    liver_csv = tmp_path / "Liver.train.v2.csv"
    gene_tss_csv = tmp_path / "gene_tss.v2.csv"

    with liver_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "common_variant_analysis_signal_id",
                "chr",
                "credible_set_variants_in_1_based_B37",
                "genes",
                "genes_with_zero_expression",
                "true_gene",
                "eqtl_tissue",
                "category",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "common_variant_analysis_signal_id": "signal-1",
                "chr": "chr1",
                "credible_set_variants_in_1_based_B37": "1-101-A-T,1-103-G-C",
                "genes": "geneA,geneB",
                "genes_with_zero_expression": "geneA,geneB",
                "true_gene": "geneA",
                "eqtl_tissue": "Liver eQTLs",
                "category": "train",
            }
        )

    gene_tss_csv.write_text("gene_id,TSS_B37\n" "geneA,151\n" "geneB,171\n")

    genome_bases = list("C" * 220)
    genome_bases[100] = "A"  # 1-based 101
    genome_bases[102] = "G"  # 1-based 103
    genome_bases[150] = "T"  # geneA TSS
    genome_bases[170] = "G"  # geneB TSS
    genome = FixedGenome("".join(genome_bases))

    records = build_liver_llm_sequence_records(
        liver_csv,
        gene_tss_csv,
        genome,
        variant_region_len=10,
        gene_region_len=12,
    )

    assert len(records) == 4
    assert {record["gene_id"] for record in records} == {"geneA", "geneB"}
    assert {record["variant_id"] for record in records} == {
        "chr1-101-A-T",
        "chr1-103-G-C",
    }
    assert all(len(record["ref_sequence"]) == 22 for record in records)
    assert all(len(record["alt_sequence"]) == 22 for record in records)
    assert all(record["variant_offset_in_sequence"] == 5 for record in records)
    assert all(record["gene_tss_offset_in_sequence"] == 16 for record in records)

    first = next(
        record
        for record in records
        if record["variant_id"] == "chr1-101-A-T" and record["gene_id"] == "geneA"
    )
    assert first["ref_sequence"][5] == "A"
    assert first["alt_sequence"][5] == "T"
    assert first["ref_sequence"][:10] == "CCCCCACGCC"
    assert first["ref_sequence"][16] == "T"

    long_records = to_long_records(records)
    assert len(long_records) == 8
    assert {record["allele_type"] for record in long_records} == {"ref", "alt"}


def test_write_records_supports_tsv_gz(tmp_path):
    out_path = tmp_path / "records.tsv.gz"
    write_records(
        out_path,
        [
            {
                "variant_id": "chr1-10-A-T",
                "gene_id": "geneA",
                "ref_sequence": "AAAA",
                "alt_sequence": "TTTT",
            }
        ],
    )

    with gzip.open(out_path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert rows == [
        {
            "variant_id": "chr1-10-A-T",
            "gene_id": "geneA",
            "ref_sequence": "AAAA",
            "alt_sequence": "TTTT",
        }
    ]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    module_globals = globals()
    for name, obj in sorted(module_globals.items()):
        if name.startswith("test_") and callable(obj):
            if "tmp_path" in obj.__code__.co_varnames[:obj.__code__.co_argcount]:
                suite.addTest(unittest.FunctionTestCase(lambda obj=obj: _run_with_tmp_path(obj)))
            else:
                suite.addTest(unittest.FunctionTestCase(obj))
    return suite


def _run_with_tmp_path(test_func):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_func(Path(tmp_dir))


if __name__ == "__main__":
    unittest.main()
