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
        "expression_delta", position_mode=position_mode, show_tf=True,
        attribution_threshold=1, attribution_threshold_mode="absolute",
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


def _profile_ranking_ism(tmp_path, *, reverse=False):
    import numpy as np

    probabilities = np.array([
        [0.34, 0.60, 0.03, 0.03],
        [0.01, 0.80, 0.18, 0.01],
        [0.04, 0.16, 0.75, 0.05],
    ])
    path = _meme_file(tmp_path)
    with path.open("a", encoding="utf-8") as out:
        out.write("\nMOTIF ProfileMatch\nletter-probability matrix: alength= 4 w= 3 nsites= 20\n")
        for row in probabilities:
            out.write(" ".join(map(str, row)) + "\n")

    coordinates = CoordinateMap([CoordinateSegment(0, 6, "genome", "chr3", 101, 107, "-")])
    sequence = AnnotatedSequence("NTCGTN" if reverse else "NACGAN", coordinate_map=coordinates)
    ism = ISM(sequence, center=2, region=(1, 5), condition="test")
    ism.tf_annotate(path, p_value_threshold=0.04)
    path.unlink()
    start = 2 if reverse else 1
    oriented = probabilities[::-1, ::-1] if reverse else probabilities
    motif_scores = np.log2((0.9999 * oriented + 0.000025) / 0.25)
    bases = "ACGT"
    effects = []
    for position, ref, alt in ism.variants.select("position", "ref", "alt").iter_rows():
        if start <= position < start + 3:
            row = motif_scores[position - start]
            effects.append(float(row[bases.index(alt)] - row[bases.index(ref)]))
        else:
            effects.append(-4.0)
    ism.variants = ism.variants.with_columns(pl.Series("expression_delta", effects))
    return ism


@pytest.mark.parametrize("reverse", [False, True])
def test_profile_agreement_outranks_better_sequence_match(tmp_path, reverse):
    ism = _profile_ranking_ism(tmp_path, reverse=reverse)
    ranked = ism.rank_tf_hits(
        "expression_delta", attribution_threshold=0.1, attribution_threshold_mode="absolute"
    )

    assert ranked.select("tf", "strand", "peak_start", "peak_end", "peak_rank").rows() == [
        ("ProfileMatch", "-" if reverse else "+", 1, 5, 1),
        ("Example", "-" if reverse else "+", 1, 5, 2),
    ]
    assert ranked["p_value"].to_list() == pytest.approx([2 / 64, 1 / 64])
    assert ranked["profile_similarity"].to_list() == pytest.approx([1.0, 0.734697], abs=1e-6)
    attribution = ism.position_attributions("expression_delta")["attribution"].abs().sum()
    assert ranked["peak_coverage"].to_list() == pytest.approx([1 - 3 / attribution] * 2)

    only_profile_match = ism.rank_tf_hits(
        "expression_delta", attribution_threshold=0.1, attribution_threshold_mode="absolute",
        min_profile_similarity=0.9
    )
    assert only_profile_match["tf"].to_list() == ["ProfileMatch"]
    assert ism.rank_tf_hits(
        "expression_delta", attribution_threshold=0.1, attribution_threshold_mode="absolute",
        min_peak_coverage=0.95
    ).is_empty()


def test_negative_profile_direction_preserves_sign(tmp_path):
    ism = _profile_ranking_ism(tmp_path)
    ism.variants = ism.variants.with_columns(-pl.col("expression_delta"))
    options = dict(
        attribution_threshold=0.1, attribution_threshold_mode="absolute", min_profile_similarity=0.9
    )
    assert ism.rank_tf_hits("expression_delta", **options).is_empty()

    for direction in ("negative", "either"):
        ranked = ism.rank_tf_hits("expression_delta", profile_direction=direction, **options)
        assert ranked["tf"].to_list() == ["ProfileMatch"]
        assert ranked["profile_similarity"].to_list() == pytest.approx([-1.0])


def test_incomplete_position_splits_peaks_and_ranks_each_one(tmp_path):
    ism = _profile_ranking_ism(tmp_path)
    ism.variants = ism.variants.with_columns(
        pl.when((pl.col("position") == 2) & (pl.col("alt") == "A"))
        .then(None)
        .otherwise(pl.col("expression_delta"))
        .alias("expression_delta")
    )

    ranked = ism.rank_tf_hits(
        "expression_delta", attribution_threshold=0.1, attribution_threshold_mode="absolute"
    )
    assert ranked.filter(pl.col("tf") == "ProfileMatch").select(
        "peak_start", "peak_end", "peak_rank"
    ).sort("peak_start").rows() == [(1, 2, 1), (3, 5, 1)]


@pytest.mark.parametrize(
    ("position_mode", "strand", "motif_bounds"),
    [("region", "+", (-0.5, 2.5)), ("genomic", "-", (102.5, 105.5))],
)
def test_attribution_plot_can_select_motifs_by_profile_similarity(
    tmp_path, position_mode, strand, motif_bounds
):
    from matplotlib import pyplot as plt

    pytest.importorskip("logomaker")
    ism = _profile_ranking_ism(tmp_path)
    options = dict(
        show_tf=True, attribution_threshold=0.1, attribution_threshold_mode="absolute",
        max_motif_hits=1, position_mode=position_mode,
    )
    fig_legacy, (_, motif_legacy) = ism.plot_attribution("expression_delta", **options)
    fig_profile, (_, motif_profile) = ism.plot_attribution(
        "expression_delta", min_profile_similarity=0.4, **options
    )
    try:
        assert [label.get_text() for label in motif_legacy.texts] == [f"Example ({strand})"]
        assert [label.get_text() for label in motif_profile.texts] == [f"ProfileMatch ({strand})"]
        fig_profile.canvas.draw()
        (logo_axis,) = motif_profile.child_axes
        bbox = motif_profile.transData.inverted().transform_bbox(logo_axis.bbox)
        assert (bbox.xmin, bbox.xmax) == pytest.approx(motif_bounds)
    finally:
        plt.close(fig_legacy)
        plt.close(fig_profile)


@pytest.fixture
def quantile_ism(tmp_path):
    ism = _ism()
    ism.tf_annotate(_meme_file(tmp_path), p_value_threshold=1 / 64)
    ism.variants = ism.variants.with_columns(
        (-4.0 * pl.col("position")).alias("expression_delta")
    )
    return ism


@pytest.mark.parametrize(("quantile", "start"), [(0.7, 6), (0.9, 7), (None, 7)])
def test_attribution_peak_quantiles(quantile_ism, quantile, start):
    ranked = quantile_ism.rank_tf_hits("expression_delta", attribution_threshold=quantile)
    assert ranked.select("peak_start", "peak_end").rows() == [(start, 8)]


def test_attribution_threshold_validation(quantile_ism):
    with pytest.raises(ValueError, match="quantile between 0 and 1"):
        quantile_ism.rank_tf_hits("expression_delta", attribution_threshold=1.5)
    ranked = quantile_ism.rank_tf_hits(
        "expression_delta", attribution_threshold=1.5, attribution_threshold_mode="absolute"
    )
    assert ranked.select("peak_start", "peak_end").sort("peak_start").rows() == [(1, 4), (5, 8)]


def test_attribution_logo_accepts_quantile_with_profile_ranking(quantile_ism):
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    pytest.importorskip("logomaker")
    fig, (attribution_ax, motif_ax) = quantile_ism.plot_attribution(
        "expression_delta", show_tf=True, attribution_threshold=0.7, min_profile_similarity=0.9
    )
    try:
        frames = [patch for patch in attribution_ax.patches if isinstance(patch, Rectangle)]
        assert [(frame.get_x(), frame.get_width()) for frame in frames] == [(4.5, 2.0)]
        assert [label.get_text() for label in motif_ax.texts] == ["Example (-)"]
    finally:
        plt.close(fig)
