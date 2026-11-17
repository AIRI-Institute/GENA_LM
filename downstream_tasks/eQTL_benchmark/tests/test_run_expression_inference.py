import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_expression_inference import (
    batched,
    build_output_rows,
    flatten_pairs_for_inference,
    load_sequence_pairs,
    read_description,
    sequence_record_to_pair,
    validate_sequence,
)
from liver_llm_sequences import (
    build_alt_variant_region,
    build_liver_llm_sequence_records,
    fetch_centered_sequence,
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


def test_validate_sequence_accepts_acgtn_and_uppercases():
    assert validate_sequence("acgtn", "seq") == "ACGTN"


def test_validate_sequence_rejects_bad_bases():
    with unittest.TestCase().assertRaisesRegex(ValueError, "invalid DNA bases"):
        validate_sequence("ACGTX", "seq")


def test_read_description_loads_text_file_and_strips_edges():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "description.txt"
        path.write_text("\nHepG2 RNA-seq description\n", encoding="utf-8")

        assert read_description(path) == "HepG2 RNA-seq description"


def test_read_description_rejects_empty_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "description.txt"
        path.write_text("  \n", encoding="utf-8")

        with unittest.TestCase().assertRaisesRegex(ValueError, "Description file is empty"):
            read_description(path)


def test_load_sequence_pairs_and_flatten():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "pairs.tsv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                delimiter="\t",
                fieldnames=["variant_id", "gene_id", "ref_sequence", "alt_sequence"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "variant_id": "chr1-10-A-T",
                    "gene_id": "geneA",
                    "ref_sequence": "ACGT",
                    "alt_sequence": "TCGT",
                }
            )

        pairs = load_sequence_pairs(path)
        sequences, keys = flatten_pairs_for_inference(pairs)

    assert len(pairs) == 1
    assert pairs[0]["metadata"] == {"variant_id": "chr1-10-A-T", "gene_id": "geneA"}
    assert sequences == ["ACGT", "TCGT"]
    assert keys == [(0, "ref"), (0, "alt")]


def test_build_output_rows_adds_ref_alt_and_delta():
    pairs = [
        {
            "pair_index": 0,
            "metadata": {"variant_id": "v1", "gene_id": "g1"},
            "ref_sequence": "AAAA",
            "alt_sequence": "TTTT",
        }
    ]
    rows = build_output_rows(
        pairs,
        prediction_by_key={(0, "ref"): 1.25, (0, "alt"): 2.0},
        description_name="HepG2",
    )

    assert rows == [
        {
            "variant_id": "v1",
            "gene_id": "g1",
            "description_name": "HepG2",
            "ref_expression": 1.25,
            "alt_expression": 2.0,
            "delta_alt_minus_ref": 0.75,
        }
    ]


def test_variant_substitution_works_for_snv_and_indels():
    assert build_alt_variant_region("CCCCACCCCC", 4, "A", "T", 10) == "CCCCTCCCCC"

    insertion = build_alt_variant_region("CCCCACCCCC", 4, "A", "ATG", 10)
    deletion = build_alt_variant_region("CCCCATCCCCGG", 4, "AT", "A", 10)

    assert insertion == "CCCCATGCCC"
    assert deletion == "CCCCACCCCG"
    assert len(insertion) == 10
    assert len(deletion) == 10


def test_fetch_centered_variant_and_gene_tss_regions_from_genome():
    genome_bases = list("C" * 220)
    genome_bases[100] = "A"  # variant at 1-based 101
    genome_bases[150] = "T"  # gene TSS at 1-based 151
    genome = FixedGenome("".join(genome_bases))

    variant_region, variant_start0 = fetch_centered_sequence(genome, "chr1", 101, 10)
    gene_region, gene_start0 = fetch_centered_sequence(genome, "chr1", 151, 12)

    assert variant_start0 == 95
    assert gene_start0 == 144
    assert variant_region == "CCCCCA CCCC".replace(" ", "")
    assert gene_region == "CCCCCCTCCCCC"
    assert variant_region[5] == "A"
    assert gene_region[6] == "T"


def test_direct_generation_concatenates_variant_and_gene_regions_correctly():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
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
                    "credible_set_variants_in_1_based_B37": "1-101-A-T",
                    "genes": "geneA",
                    "genes_with_zero_expression": "geneA",
                    "true_gene": "geneA",
                    "eqtl_tissue": "Liver eQTLs",
                    "category": "train",
                }
            )

        gene_tss_csv.write_text("gene_id,TSS_B37\n" "geneA,151\n")

        genome_bases = list("C" * 220)
        genome_bases[100] = "A"  # variant at 1-based 101
        genome_bases[150] = "G"  # gene TSS at 1-based 151
        genome = FixedGenome("".join(genome_bases))

        records = build_liver_llm_sequence_records(
            liver_csv,
            gene_tss_csv,
            genome,
            variant_region_len=10,
            gene_region_len=12,
        )

    assert len(records) == 1
    record = records[0]
    assert record["variant_region_start_0based"] == 95
    assert record["gene_region_start_0based"] == 144
    assert record["variant_offset_in_sequence"] == 5
    assert record["gene_tss_offset_in_sequence"] == 16
    assert record["ref_sequence"] == "CCCCCACCCC" + "CCCCCCGCCCCC"
    assert record["alt_sequence"] == "CCCCCTCCCC" + "CCCCCCGCCCCC"
    assert record["ref_sequence"][5] == "A"
    assert record["alt_sequence"][5] == "T"
    assert record["ref_sequence"][16] == "G"
    assert record["alt_sequence"][16] == "G"


def test_batched_groups_items_without_dropping_tail():
    batches = list(batched(({"x": i} for i in range(5)), batch_size=2))

    assert batches == [[{"x": 0}, {"x": 1}], [{"x": 2}, {"x": 3}], [{"x": 4}]]


def test_sequence_record_to_pair_strips_sequences_from_metadata():
    pair = sequence_record_to_pair(
        {
            "variant_id": "v1",
            "gene_id": "g1",
            "ref_sequence": "AAAA",
            "alt_sequence": "TTTT",
        },
        pair_index=7,
    )

    assert pair == {
        "pair_index": 7,
        "metadata": {"variant_id": "v1", "gene_id": "g1"},
        "ref_sequence": "AAAA",
        "alt_sequence": "TTTT",
    }


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, obj in sorted(globals().items()):
        if name.startswith("test_") and callable(obj):
            suite.addTest(unittest.FunctionTestCase(obj))
    return suite


if __name__ == "__main__":
    unittest.main()
