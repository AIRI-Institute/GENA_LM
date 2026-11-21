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


def test_position_attribution_signed_mean_and_incomplete_coverage():
    ism = ISM(
        AnnotatedSequence("ACGTN"), center=0, region=(0, 5), condition="test", n_deletions=5
    )
    ism.variants = ism.variants.with_columns(
        pl.when(pl.col("mutation_type") == "deletion").then(99.0)
        .when((pl.col("position") == 0) & (pl.col("alt") == "C")).then(-8.0)
        .when((pl.col("position") == 0) & (pl.col("alt") == "T")).then(0.0)
        .when((pl.col("position") == 2) & (pl.col("alt") == "C")).then(float("nan"))
        .when(pl.col("position") < 3).then(4.0)
        .otherwise(None)
        .alias("expression_delta")
    )

    assert ism.position_attributions("expression_delta").rows() == [
        (0, 0, "A", 3, 1.0),
        (1, 1, "C", 3, -3.0),
        (2, 2, "G", 2, None),
        (3, 3, "T", 0, None),
        (4, 4, "N", 0, None),
    ]


@pytest.mark.parametrize(
    ("position_mode", "bounds", "highlight_lefts", "glyphs"),
    [
        (
            "region", (-0.5, 6.5), [-0.5, 4.5],
            [(0, "A", 3.0), (1, "C", -3.0), (5, "G", -1.5)],
        ),
        (
            "genomic", (101.5, 108.5), [102.5, 106.5],
            [(103, "C", -1.5), (107, "G", -3.0), (108, "T", 3.0)],
        ),
    ],
)
def test_attribution_logo_shades_matching_motif_bases(
    tmp_path, position_mode, bounds, highlight_lefts, glyphs
):
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch, Rectangle

    from gena_expression.ism.plotting import _ALT_COLORS

    pytest.importorskip("logomaker")
    ism = _ism(CoordinateMap([CoordinateSegment(0, 9, "genome", "chr3", 101, 110, "-")]))
    path = _meme_file(tmp_path)
    ism.tf_annotate(path, p_value_threshold=1 / 64)
    path.unlink()  # Logos use the matrices that were loaded for the scan.
    ism.variants = ism.variants.with_columns(
        pl.col("position").replace_strict(
            {1: -4.0, 2: 4.0, 6: 2.0}, default=0.0
        ).alias("expression_delta")
    )

    fig, (attribution_ax, motif_ax) = ism.plot_attribution(
        "expression_delta", position_mode=position_mode, show_tf=True, attribution_threshold=1
    )
    try:
        fig.canvas.draw()
        assert attribution_ax.get_xlim() == motif_ax.get_xlim() == bounds
        assert attribution_ax.get_shared_x_axes().joined(attribution_ax, motif_ax)
        shaded_regions = [patch for patch in attribution_ax.patches if isinstance(patch, Rectangle)]
        assert [patch.get_x() for patch in shaded_regions] == highlight_lefts

        reference_letters = []
        for patch in attribution_ax.patches:
            if isinstance(patch, PathPatch):
                box = patch.get_path().get_extents()
                base = next(
                    base for base, color in _ALT_COLORS.items() if patch.get_facecolor() == to_rgba(color)
                )
                sign = 1 if box.y0 + box.y1 > 0 else -1
                height = sign * box.height
                reference_letters.append((round((box.x0 + box.x1) / 2), base, height))
        assert sorted(reference_letters) == [
            (position, base, pytest.approx(height)) for position, base, height in glyphs
        ]

        assert len(motif_ax.child_axes) == 2
        letters = []
        for child in motif_ax.child_axes:
            spans = [patch for patch in child.patches if isinstance(patch, Rectangle)]
            assert len(spans) == 1
            # Compare in display coordinates to verify that the inset's fill
            # occupies exactly the same horizontal interval as its upper fill.
            child_span = spans[0].get_transform().transform([[0, 0], [1, 1]])[:, 0]
            assert any(
                child_span == pytest.approx(patch.get_transform().transform([[0, 0], [1, 1]])[:, 0])
                for patch in shaded_regions
            )
            colored = [patch for patch in child.patches if isinstance(patch, PathPatch)]
            letters.append("".join(
                next(base for base, color in _ALT_COLORS.items() if patch.get_facecolor() == to_rgba(color))
                for patch in sorted(colored, key=lambda patch: patch.get_path().get_extents().x0)
            ))
        assert letters == ["ACG", "CGT"]
    finally:
        plt.close(fig)


def test_logo_without_motif_overlap_and_zero_effects(tmp_path):
    from matplotlib import pyplot as plt

    pytest.importorskip("logomaker")
    ism = _ism()
    ism.tf_annotate(_meme_file(tmp_path), p_value_threshold=1 / 64)
    ism.variants = ism.variants.with_columns(pl.lit(0.0).alias("expression_delta"))
    fig, (attribution_ax, motif_ax) = ism.plot_attribution("expression_delta", show_tf=True)
    try:
        assert not attribution_ax.patches
        assert not motif_ax.child_axes
        assert [text.get_text() for text in motif_ax.texts] == ["No motif hits overlap high attribution"]
    finally:
        plt.close(fig)
