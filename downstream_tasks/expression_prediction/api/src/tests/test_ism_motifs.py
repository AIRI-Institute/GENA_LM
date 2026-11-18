"""Regression tests for reference motif annotation and coordinate-aware plotting."""

import matplotlib
import polars as pl
import pytest

from gena_expression import AnnotatedSequence, CoordinateMap, CoordinateSegment, ISM

matplotlib.use("Agg")


def _meme_file(tmp_path, background="A 0.25 C 0.25 G 0.25 T 0.25"):
    motif_file = tmp_path / "motifs.meme"
    motif_file.write_text(
        "MEME version 4\n\n"
        "ALPHABET= ACGT\n"
        "strands: +\n\n"
        f"Background letter frequencies\n{background}\n\n"
        "MOTIF\tExample.H14CORE.0.P.A\n"
        "letter-probability matrix: alength= 4 w= 3 nsites= 20\n"
        "1 0 0 0\n"
        "0 1 0 0\n"
        "0 0 1 0\n",
        encoding="utf-8",
    )
    return motif_file


def _ism(coordinate_map=None):
    sequence = AnnotatedSequence("NACGNCGTN", name="test", coordinate_map=coordinate_map)
    return ISM(sequence, center=1, region=(1, 8), condition="condition")


def test_reference_hits_include_strand_threshold_and_half_open_peak_overlap(tmp_path):
    ism = _ism()
    hits = ism.tf_annotate(
        _meme_file(tmp_path), p_value_threshold=1 / 64, peak_intervals=[(4, 5), (7, 8)]
    )
    assert hits is ism.tf_hits
    assert hits.select("tf", "start", "end", "start_in_region", "end_in_region", "strand").rows() == [
        ("Example", 1, 4, 0, 3, "+"),
        ("Example", 5, 8, 4, 7, "-"),
    ]
    assert hits.get_column("p_value").to_list() == pytest.approx([1 / 64, 1 / 64])
    assert hits.get_column("overlaps_peak").to_list() == [False, True]
    assert hits.get_column("source_start").to_list() == [None, None]
    assert "Example.H14CORE.0.P.A" in hits.get_column("motif_id").to_list()

    empty = ism.tf_annotate(_meme_file(tmp_path), p_value_threshold=0.015)
    assert empty.is_empty()
    assert empty.schema == hits.schema


def test_strand_specific_tail_accounts_for_asymmetric_background(tmp_path):
    motif_file = _meme_file(tmp_path, "A 0.5 C 0.125 G 0.125 T 0.25")
    hits = _ism().tf_annotate(motif_file, p_value_threshold=0.005)
    assert hits.select("start", "strand", "p_value").rows() == [(5, "-", 0.00390625)]
    assert hits.get_column("overlaps_peak").to_list() == [None]


def test_reverse_genomic_projection_and_shared_plot_axes(tmp_path):
    from matplotlib import pyplot as plt

    ism = _ism(CoordinateMap([CoordinateSegment(0, 9, "genome", "chr3", 101, 110, "-")]))
    hits = ism.tf_annotate(_meme_file(tmp_path), p_value_threshold=1 / 64)
    assert hits.select("source_chrom", "source_start", "source_end", "source_strand").rows() == [
        ("chr3", 106, 109, "-"),
        ("chr3", 102, 105, "+"),
    ]
    ism.variants = ism.variants.with_columns(pl.lit(1.0).alias("dummy"))
    fig_plain, axes_plain = ism.plot("dummy")
    fig_combined, axes = ism.plot("dummy", position_mode="genomic", show_tf=True)
    fig_standalone, motif_ax = ism.tf_plot()
    try:
        assert len(axes_plain) == 1
        assert len(axes) == 2
        assert axes[0].get_shared_x_axes().joined(*axes)
        assert axes[0].get_xlim() == axes[1].get_xlim() == motif_ax.get_xlim() == (102, 109)
        assert axes[1].get_xlabel() == "chr3 position (0-based)"
        assert [tick.get_text() for tick in motif_ax.get_yticklabels()] == ["Example"]
        assert [(patch.get_x(), patch.get_width()) for patch in axes[1].patches] == [(106, 3), (102, 3)]
        assert fig_combined.get_figheight() > fig_plain.get_figheight()
    finally:
        for fig in (fig_plain, fig_combined, fig_standalone):
            plt.close(fig)


def test_discontinuous_projection_keeps_local_hits(tmp_path):
    ism = _ism(
        CoordinateMap([
            CoordinateSegment(0, 2, "genome", "chr1", 101, 103, "+"),
            CoordinateSegment(2, 9, "genome", "chr1", 110, 117, "+"),
        ])
    )
    hits = ism.tf_annotate(_meme_file(tmp_path), p_value_threshold=1 / 64)
    assert hits.select("start", "end", "source_start").rows() == [(1, 4, None), (5, 8, 113)]
    with pytest.raises(ValueError, match="no contiguous genomic projection"):
        ism.tf_plot()


def test_unannotated_and_empty_motif_panels(tmp_path):
    from matplotlib import pyplot as plt

    ism = _ism()
    with pytest.raises(ValueError, match="Call tf_annotate"):
        ism.tf_plot(position_mode="region")
    ism.tf_annotate(_meme_file(tmp_path))
    fig, ax = ism.tf_plot(position_mode="region")
    try:
        assert not ax.patches
        assert [text.get_text() for text in ax.texts] == ["No motif hits"]
    finally:
        plt.close(fig)


def test_invalid_annotation_options_and_meme_schema(tmp_path):
    ism = _ism()
    path = _meme_file(tmp_path)
    with pytest.raises(ValueError, match="p_value_threshold"):
        ism.tf_annotate(path, p_value_threshold=float("nan"))
    with pytest.raises(ValueError, match="Peak intervals"):
        ism.tf_annotate(path, peak_intervals=[(4, 4)])

    path.write_text("MEME version 4\n\nALPHABET= ACGT\nMOTIF missing_matrix\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing letter-probability matrix"):
        ism.tf_annotate(path)

    path = _meme_file(tmp_path)
    path.write_text(path.read_text().replace("strands: +", "strands: + ?"), encoding="utf-8")
    with pytest.raises(ValueError, match="MEME strands"):
        ism.tf_annotate(path)
    path.write_text(path.read_text().replace("ALPHABET= ACGT", "ALPHABET= ACGU"), encoding="utf-8")
    with pytest.raises(ValueError, match="MEME DNA alphabet"):
        ism.tf_annotate(path)
