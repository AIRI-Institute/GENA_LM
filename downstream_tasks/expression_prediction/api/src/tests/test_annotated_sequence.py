"""Compatibility tests for sequence operations and plasmid loading."""

import pytest

from gena_expression import AnnotatedSequence, Feature, PlasmidRecord


def test_existing_sequence_operations_keep_their_contracts():
    sequence = AnnotatedSequence(
        "acgt", name="example", features=[Feature("site", 1, 3, strand="+")]
    )

    assert sequence.to_fasta() == ">example\nACGT\n"
    assert sequence.to_dict()["features"][0]["start"] == 1
    assert sequence[1:3].sequence == "CG"
    assert sequence.reverse_complement().features[0].strand == "-"
    assert sequence.sequence == "ACGT"
    assert AnnotatedSequence("").sequence == ""
    assert Feature("empty", 2, 2).length() == 0
    assert AnnotatedSequence("X").sequence == "X"


@pytest.mark.parametrize(
    ("location", "strand", "tss"),
    [("2..4", "+", 1), ("complement(2..4)", "-", 3)],
)
@pytest.mark.parametrize("use_fasta", [False, True])
def test_plasmid_reads_reporter_coordinates(tmp_path, location, strand, tss, use_fasta):
    genbank = tmp_path / "external.gbk"
    genbank.write_text(
        "LOCUS       plasmid 8 bp DNA circular\n"
        "FEATURES             Location/Qualifiers\n"
        f"     CDS             {location}\n"
        '                     /gene="luc2"\n'
        "ORIGIN\n"
        "        1 acgtacgt\n"
        "//\n",
        encoding="utf-8",
    )
    fasta = tmp_path / "external.fa"
    fasta.write_text(">plasmid\nACGTACGT\n", encoding="utf-8")

    plasmid = PlasmidRecord.from_files(genbank, fasta if use_fasta else None)
    annotated = plasmid.annotated_sequence()

    assert plasmid.name == "plasmid"
    assert plasmid.features["luc2"] == (2, 4)
    assert plasmid.reporter_tss_index() == tss
    assert annotated.sequence == "ACGTACGT"
    assert annotated.metadata["record_type"] == "plasmid"
    reporter = annotated.features[0]
    assert (reporter.name, reporter.start, reporter.end) == ("luc2", 1, 4)
    assert (reporter.strand, reporter.source) == (strand, "genbank")
