"""Add nearby, same-strand transcript TSS features to sequence pairs.

This is a standalone helper for ``gena_expression``. It does not modify the
package itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from numbers import Integral
from typing import Any
import warnings

from gena_expression.sequences import AnnotatedSequence, Feature, SequencePair


def _tss_base(feature: Feature, sequence_length: int) -> int:
    """Return the sequence-local, zero-based base representing a TSS."""
    if feature.strand == "+":
        tss = feature.start
    elif feature.strand == "-":
        tss = feature.end - 1
    else:
        raise ValueError(
            f"Transcript {feature.name!r} has strand {feature.strand!r}; "
            "expected '+' or '-'."
        )

    if not 0 <= tss < sequence_length:
        raise ValueError(
            f"Transcript {feature.name!r} has local TSS {tss}, outside "
            f"sequence length {sequence_length}."
        )
    return tss


def _contains_unclipped_tss(
    sequence: AnnotatedSequence,
    transcript: Feature,
) -> bool:
    """Return whether an overlapping transcript's real TSS is in the window."""
    metadata = transcript.metadata or {}
    genomic_start = metadata.get("genomic_start")
    genomic_end = metadata.get("genomic_end")
    if genomic_start is None or genomic_end is None:
        return True

    local_tss = _tss_base(transcript, len(sequence))
    chrom = metadata.get("chrom")
    coordinate_map = sequence.coordinate_map
    has_genome_mapping = any(
        segment.source == "genome" for segment in coordinate_map.segments
    )
    if has_genome_mapping:
        # Feature strands and local coordinates change after reverse
        # complementation or slicing. The source map remains tied to the
        # genome, so test both possible transcript endpoints and let the
        # feature's current strand identify which endpoint is its TSS.
        genome_chroms = {
            segment.chrom
            for segment in coordinate_map.segments
            if segment.source == "genome"
        }
        endpoint_positions: set[int | None] = set()
        for endpoint in (int(genomic_start), int(genomic_end) - 1):
            mapped = coordinate_map.source_to_seq(
                "genome",
                endpoint,
                chrom=None if chrom is None else str(chrom),
            )
            # FASTA and annotation sources sometimes spell the same chromosome
            # differently (for example "11" versus "chr11"). Falling back is
            # safe when the sequence map contains only one genomic chromosome.
            if mapped is None and len(genome_chroms) == 1:
                mapped = coordinate_map.source_to_seq("genome", endpoint)
            endpoint_positions.add(mapped)
        return local_tss in endpoint_positions

    # Compatibility fallback for manually constructed sequences without a
    # genome coordinate map. This covers the original, forward-oriented form.
    genome_interval = next(
        (
            feature
            for feature in sequence.features
            if feature.type.casefold() == "genome_interval"
            and (feature.metadata or {}).get("start") is not None
        ),
        None,
    )
    if genome_interval is None:
        return True

    window_start = int((genome_interval.metadata or {})["start"])
    genomic_tss = (
        int(genomic_start)
        if transcript.strand == "+"
        else int(genomic_end) - 1
    )
    return genomic_tss - window_start == local_tss


def _transcript_key(feature: Feature) -> tuple[Any, ...]:
    """Return transcript identity stable across reference and alternative."""
    metadata: Mapping[str, Any] = feature.metadata or {}
    attributes = metadata.get("attributes")
    if isinstance(attributes, Mapping) and attributes.get("transcript_id"):
        return ("transcript_id", str(attributes["transcript_id"]))

    return (
        "coordinates",
        metadata.get("chrom"),
        metadata.get("genomic_start"),
        metadata.get("genomic_end"),
        feature.name,
        feature.strand,
        feature.source,
    )


def _candidate_transcripts(
    sequence: AnnotatedSequence,
    *,
    transcript_type: str,
) -> list[tuple[Feature, int]]:
    """Return transcripts whose actual TSS lies inside the sequence."""
    transcripts = [
        feature
        for feature in sequence.features
        if feature.type.casefold() == transcript_type.casefold()
    ]
    candidates = [
        (feature, _tss_base(feature, len(sequence)))
        for feature in transcripts
        if _contains_unclipped_tss(sequence, feature)
    ]
    if candidates:
        return candidates

    detail = (
        "no transcript features were found"
        if not transcripts
        else "all transcript TSSs lie outside the sequence window"
    )
    raise ValueError(f"Sequence {sequence.name!r}: {detail}.")


def _nearest_anchor(
    candidates: Sequence[tuple[Feature, int]],
    *,
    variant_start: int,
    require_unique_side: bool,
) -> tuple[str, int | None, int]:
    """Return the nearest TSS strand, optional side, and distance."""
    nearest_distance = min(
        abs(tss - variant_start) for _, tss in candidates
    )
    nearest = [
        (feature, tss)
        for feature, tss in candidates
        if abs(tss - variant_start) == nearest_distance
    ]
    nearest_strands = {feature.strand for feature, _ in nearest}
    if len(nearest_strands) != 1:
        summary = ", ".join(
            f"{feature.name}@{tss}({feature.strand})"
            for feature, tss in nearest
        )
        raise ValueError(
            "Equally near transcript TSSs occur on different strands: "
            f"{summary}."
        )

    nearest_sides = {
        _position_side(tss, variant_start=variant_start)
        for _, tss in nearest
    }
    if require_unique_side and len(nearest_sides) != 1:
        summary = ", ".join(
            f"{feature.name}@{tss}({feature.strand})"
            for feature, tss in nearest
        )
        raise ValueError(
            "Equally near transcript TSSs occur on opposite sides of the "
            f"variant: {summary}."
        )
    side = (
        next(iter(nearest_sides))
        if len(nearest_sides) == 1
        else None
    )
    return (
        next(iter(nearest_strands)),
        side,
        nearest_distance,
    )


def _position_side(position: int, *, variant_start: int) -> int:
    """Return -1, 0, or 1 for left, coincident, or right of the variant."""
    if position < variant_start:
        return -1
    if position > variant_start:
        return 1
    return 0


def _unique_same_strand_transcripts(
    sequence: AnnotatedSequence,
    *,
    variant_start: int,
    transcript_type: str,
    max_distance: int,
    side_mode: str,
) -> list[tuple[Feature, int, int, bool]]:
    """Return every eligible unique TSS in deterministic distance order."""
    candidates = _candidate_transcripts(
        sequence,
        transcript_type=transcript_type,
    )
    strand, side, nearest_distance = _nearest_anchor(
        candidates,
        variant_start=variant_start,
        require_unique_side=side_mode == "nearest_side",
    )
    if nearest_distance > max_distance:
        raise ValueError(
            f"Nearest transcript TSS is {nearest_distance} bp from the "
            f"variant, beyond max_distance={max_distance}."
        )

    same_strand = [
        (
            feature,
            tss,
            abs(tss - variant_start),
            abs(tss - variant_start) == nearest_distance,
        )
        for feature, tss in candidates
        if feature.strand == strand
        and (
            side_mode == "both"
            or _position_side(tss, variant_start=variant_start) == side
        )
        and abs(tss - variant_start) <= max_distance
    ]
    same_strand.sort(
        key=lambda item: (
            item[2],
            item[1],
            repr(_transcript_key(item[0])),
        )
    )

    # Several transcript isoforms may start at the same base. Keep one
    # deterministic representative so every added TSS coordinate is unique.
    unique: dict[int, tuple[Feature, int, int, bool]] = {}
    for item in same_strand:
        unique.setdefault(item[1], item)
    return list(unique.values())


def _cluster_nearby_tss(
    selected: Sequence[tuple[Feature, int, int, bool]],
    *,
    variant_start: int,
    min_tss_dist: int,
) -> list[tuple[tuple[Feature, ...], tuple[int, ...], int, int, bool]]:
    """Merge nearby TSS coordinates and position each cluster at its mean.

    Clustering is transitive along sorted coordinates. For example, positions
    100, 110, and 120 form one cluster when ``min_tss_dist=11``, even though
    the first and last positions are 20 bases apart.
    """
    by_position = sorted(selected, key=lambda item: item[1])
    groups: list[list[tuple[Feature, int, int, bool]]] = []
    for item in by_position:
        if (
            groups
            and item[1] - groups[-1][-1][1] < min_tss_dist
        ):
            groups[-1].append(item)
        else:
            groups.append([item])

    clustered = []
    for group in groups:
        positions = tuple(item[1] for item in group)
        mean_tss = int(sum(positions) / len(positions))
        clustered.append(
            (
                tuple(item[0] for item in group),
                positions,
                mean_tss,
                abs(mean_tss - variant_start),
                any(item[3] for item in group),
            )
        )
    clustered.sort(
        key=lambda item: (
            item[3],
            item[2],
            tuple(repr(_transcript_key(feature)) for feature in item[0]),
        )
    )
    return clustered


def _matching_transcript(
    sequence: AnnotatedSequence,
    transcript: Feature,
    *,
    transcript_type: str,
) -> tuple[Feature, int]:
    """Find the same biological transcript in the other allele."""
    key = _transcript_key(transcript)
    matches = [
        feature
        for feature in sequence.features
        if feature.type.casefold() == transcript_type.casefold()
        and _transcript_key(feature) == key
    ]
    if not matches:
        raise ValueError(
            f"Sequence {sequence.name!r} has no usable match for transcript "
            f"{transcript.name!r}; key={key!r}."
        )

    positioned = [
        (feature, _tss_base(feature, len(sequence)))
        for feature in matches
    ]
    anchors = {(tss, feature.strand) for feature, tss in positioned}
    if len(anchors) != 1:
        raise ValueError(
            f"Sequence {sequence.name!r} has ambiguous matches for "
            f"transcript {transcript.name!r}."
        )
    return min(positioned, key=lambda item: repr(_transcript_key(item[0])))


def _tss_center(feature: Feature) -> int:
    """Return the center base of a TSS feature."""
    return feature.start + (feature.end - feature.start) // 2


def _remove_tss_at(
    sequence: AnnotatedSequence,
    *,
    tss: int,
) -> AnnotatedSequence:
    """Remove every TSS feature centered at one sequence position."""
    return sequence.map_features(
        lambda feature: (
            None
            if feature.type.casefold() == "tss"
            and _tss_center(feature) == tss
            else feature
        )
    )


def _remove_generated_features(
    sequence: AnnotatedSequence,
    *,
    feature_name_prefix: str,
) -> AnnotatedSequence:
    """Remove features previously generated with the requested prefix."""
    name_prefix = f"{feature_name_prefix}_"
    return sequence.map_features(
        lambda feature: (
            None
            if feature.type.casefold() == "tss"
            and feature.source == "derived_from_transcript"
            and feature.name.startswith(name_prefix)
            else feature
        )
    )


def _window_bounds(
    *,
    tss: int,
    sequence_length: int,
    window: int,
    sequence_name: str | None,
) -> tuple[int, int]:
    """Return an exact-width interval centered on a TSS."""
    start = tss - window // 2
    end = start + window
    if start < 0 or end > sequence_length:
        raise ValueError(
            f"TSS {tss} in sequence {sequence_name!r} cannot support a "
            f"{window}-bp window inside sequence length {sequence_length}; "
            f"requested interval is [{start}, {end})."
        )
    return start, end


def _add_tss_feature(
    sequence: AnnotatedSequence,
    *,
    transcripts: Sequence[Feature],
    source_tss_positions: Sequence[int],
    tss: int,
    distance: int,
    is_nearest: bool,
    feature_name: str,
    window: int,
) -> AnnotatedSequence:
    """Return a sequence with one derived, possibly clustered TSS feature."""
    if not transcripts:
        raise ValueError("At least one transcript is required for a TSS feature.")
    start, end = _window_bounds(
        tss=tss,
        sequence_length=len(sequence),
        window=window,
        sequence_name=sequence.name,
    )
    return sequence.add_feature(
        feature_name,
        start,
        end,
        type="tss",
        strand=transcripts[0].strand,
        source="derived_from_transcript",
        metadata={
            # Keep the singular fields for compatibility with earlier output.
            "transcript_name": transcripts[0].name,
            "transcript_key": _transcript_key(transcripts[0]),
            "transcript_names": tuple(
                transcript.name for transcript in transcripts
            ),
            "transcript_keys": tuple(
                _transcript_key(transcript) for transcript in transcripts
            ),
            "source_tss_positions": tuple(source_tss_positions),
            "cluster_size": len(source_tss_positions),
            "distance_to_variant_start_bp": distance,
            "is_nearest_transcript_tss": is_nearest,
            "window_bp": window,
        },
    )


def _process_sequence_pair(
    pair: SequencePair,
    *,
    max_distance: int,
    max_tss_count: int | None,
    min_tss_dist: int,
    window: int,
    transcript_type: str,
    feature_name_prefix: str,
    side_mode: str,
    orient_to_nearest_strand: bool,
) -> SequencePair:
    """Add the requested number of usable TSSs to one sequence pair."""
    ref_variant, alt_variant = pair.variant_features()
    if ref_variant is None or alt_variant is None:
        raise ValueError(
            f"Pair {pair.ref.name!r}/{pair.alt.name!r} needs a variant "
            "feature on both alleles."
        )

    selected = _unique_same_strand_transcripts(
        pair.ref,
        variant_start=ref_variant.start,
        transcript_type=transcript_type,
        max_distance=max_distance,
        side_mode=side_mode,
    )
    clustered = _cluster_nearby_tss(
        selected,
        variant_start=ref_variant.start,
        min_tss_dist=min_tss_dist,
    )
    nearest_strand = selected[0][0].strand
    # Re-running the helper should rebuild its own annotations instead of
    # treating old one-base features as valid 501-base windows.
    ref = _remove_generated_features(
        pair.ref,
        feature_name_prefix=feature_name_prefix,
    )
    alt = _remove_generated_features(
        pair.alt,
        feature_name_prefix=feature_name_prefix,
    )
    used_alt_tss: set[int] = set()
    retained_count = 0
    skipped: list[str] = []

    for (
        ref_transcripts,
        ref_source_tss,
        ref_tss,
        distance,
        is_nearest,
    ) in clustered:
        if max_tss_count is not None and retained_count >= max_tss_count:
            break

        alt_matches = [
            _matching_transcript(
                pair.alt,
                ref_transcript,
                transcript_type=transcript_type,
            )
            for ref_transcript in ref_transcripts
        ]
        alt_transcripts = tuple(item[0] for item in alt_matches)
        alt_source_tss = tuple(item[1] for item in alt_matches)
        if any(
            transcript.strand != nearest_strand
            for transcript in alt_transcripts
        ):
            names = ", ".join(
                repr(transcript.name) for transcript in ref_transcripts
            )
            raise ValueError(
                f"Transcript cluster {names} has inconsistent "
                "reference/alternative strands."
            )
        alt_tss = int(sum(alt_source_tss) / len(alt_source_tss))
        if alt_tss in used_alt_tss:
            names = ", ".join(
                repr(transcript.name) for transcript in ref_transcripts
            )
            skipped.append(
                f"{names}: alternative TSS {alt_tss} "
                "duplicates an earlier TSS"
            )
            continue

        try:
            _window_bounds(
                tss=ref_tss,
                sequence_length=len(ref),
                window=window,
                sequence_name=ref.name,
            )
            _window_bounds(
                tss=alt_tss,
                sequence_length=len(alt),
                window=window,
                sequence_name=alt.name,
            )
        except ValueError as error:
            names = ", ".join(
                repr(transcript.name) for transcript in ref_transcripts
            )
            skipped.append(f"{names}: {error}")
            continue

        used_alt_tss.add(alt_tss)

        # Replace an old or differently sized feature at this TSS. This keeps
        # one TSS feature per coordinate and makes changing ``window`` safe.
        ref = _remove_tss_at(ref, tss=ref_tss)
        alt = _remove_tss_at(alt, tss=alt_tss)

        retained_count += 1
        feature_name = f"{feature_name_prefix}_{retained_count}"
        ref = _add_tss_feature(
            ref,
            transcripts=ref_transcripts,
            source_tss_positions=ref_source_tss,
            tss=ref_tss,
            distance=distance,
            is_nearest=is_nearest,
            feature_name=feature_name,
            window=window,
        )
        alt = _add_tss_feature(
            alt,
            transcripts=alt_transcripts,
            source_tss_positions=alt_source_tss,
            tss=alt_tss,
            distance=distance,
            is_nearest=is_nearest,
            feature_name=feature_name,
            window=window,
        )

    if retained_count == 0:
        detail = "; ".join(skipped) or "no eligible unique TSS was retained"
        raise ValueError(
            f"Pair {pair.ref.name!r}/{pair.alt.name!r}: could not add any "
            f"TSS features. {detail}"
        )

    if max_tss_count is not None and retained_count < max_tss_count:
        detail = "; ".join(skipped)
        warnings.warn(
            f"Pair {pair.ref.name!r}/{pair.alt.name!r}: requested "
            f"{max_tss_count} TSS features but added {retained_count}; only "
            f"{len(clustered)} TSS clusters were eligible."
            + (f" Skipped: {detail}" if detail else ""),
            RuntimeWarning,
            stacklevel=2,
        )
    elif skipped:
        warnings.warn(
            f"Pair {pair.ref.name!r}/{pair.alt.name!r}: skipped "
            f"{len(skipped)} unusable TSS candidate(s): {'; '.join(skipped)}",
            RuntimeWarning,
            stacklevel=2,
        )

    if orient_to_nearest_strand and nearest_strand == "-":
        ref = ref.reverse_complement()
        alt = alt.reverse_complement()

    result = SequencePair(
        ref=ref,
        alt=alt,
        variant=pair.variant,
        metadata=pair.metadata,
    )
    result.assert_compatible()
    return result


def add_same_strand_transcript_tss(
    sequence_pairs: Sequence[SequencePair],
    *,
    max_distance: int,
    max_tss_count: int | None = None,
    min_tss_dist: int = 1,
    window: int = 1,
    max_workers: int | None = None,
    chunksize: int = 16,
    transcript_type: str = "transcript",
    feature_name_prefix: str = "transcript_tss",
    side_mode: str = "both",
    orient_to_nearest_strand: bool = True,
    show_progress: bool = True,
) -> list[SequencePair]:
    """Add unique nearby TSSs sharing the nearest transcript's strand.

    For each pair, the function:

    1. finds the transcript TSS nearest to the reference variant start;
    2. uses that transcript's strand;
    3. keeps transcript TSSs on that strand and, depending on ``side_mode``,
       either both sides or only the nearest TSS's side of the variant;
    4. discards TSSs farther than ``max_distance``;
    5. merges TSS coordinates separated by less than ``min_tss_dist``;
    6. retains up to ``max_tss_count`` successfully usable TSS clusters;
    7. adds matching TSS-centered features of exactly ``window`` bases to both
       alleles;
    8. optionally reverse-complements both alleles when the selected strand is
       negative.

    Parameters
    ----------
    sequence_pairs
        Sequence pairs containing transcript and variant features.
    max_distance
        Inclusive maximum distance in base pairs from the reference variant
        start to a transcript TSS. Must be non-negative.
    max_tss_count
        Maximum number of unique TSS features added to each allele after
        clustering, ordered by distance from the variant. ``None`` keeps every
        eligible cluster. If fewer usable clusters exist, all available
        clusters are added and a warning explains the shortfall.
    min_tss_dist
        Merge TSS coordinates whose consecutive distance is strictly smaller
        than this many base pairs. Each cluster is placed at
        ``int(mean(member_positions))``. The default ``1`` only collapses
        identical coordinates.
    window
        Exact feature width in base pairs, centered on each TSS. ``1`` adds the
        original one-base feature. ``501`` adds 250 bases before the TSS, the
        TSS base, and 250 bases after it. A full window must fit inside each
        allele sequence.
    max_workers
        Number of worker processes. Pass ``0`` or ``1`` to run serially, which
        is useful for debugging and small inputs.
    chunksize
        Number of pairs sent to each process-pool task batch.
    transcript_type
        Feature type used to identify transcript annotations.
    feature_name_prefix
        Added features are named ``<prefix>_1``, ``<prefix>_2``, and so on,
        ordered by distance from the variant.
    side_mode
        ``"both"`` searches both sides of the variant on the nearest
        transcript's strand. ``"nearest_side"`` searches only the side
        containing the nearest TSS.
    orient_to_nearest_strand
        Reverse-complement both alleles when the selected strand is negative.
    show_progress
        Show a tqdm progress bar when tqdm is installed.

    Notes
    -----
    Re-running the function with the same ``feature_name_prefix`` replaces its
    previous derived TSS features. A TSS already present at a selected
    coordinate is replaced by one feature of the requested ``window`` width.
    """
    if isinstance(max_distance, bool) or not isinstance(
        max_distance, Integral
    ):
        raise TypeError("max_distance must be an integer number of base pairs.")
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    max_distance = int(max_distance)
    if max_tss_count is not None:
        if isinstance(max_tss_count, bool) or not isinstance(
            max_tss_count, Integral
        ):
            raise TypeError("max_tss_count must be an integer or None.")
        if max_tss_count < 1:
            raise ValueError("max_tss_count must be at least 1 or None.")
        max_tss_count = int(max_tss_count)
    if isinstance(min_tss_dist, bool) or not isinstance(
        min_tss_dist, Integral
    ):
        raise TypeError("min_tss_dist must be an integer number of base pairs.")
    if min_tss_dist < 0:
        raise ValueError("min_tss_dist must be non-negative.")
    min_tss_dist = int(min_tss_dist)
    if isinstance(window, bool) or not isinstance(window, Integral):
        raise TypeError("window must be an integer number of base pairs.")
    if window < 1:
        raise ValueError("window must be at least 1.")
    window = int(window)
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1.")
    if not feature_name_prefix:
        raise ValueError("feature_name_prefix must not be empty.")
    if side_mode not in {"both", "nearest_side"}:
        raise ValueError("side_mode must be 'both' or 'nearest_side'.")

    worker = partial(
        _process_sequence_pair,
        max_distance=max_distance,
        max_tss_count=max_tss_count,
        min_tss_dist=min_tss_dist,
        window=window,
        transcript_type=transcript_type,
        feature_name_prefix=feature_name_prefix,
        side_mode=side_mode,
        orient_to_nearest_strand=orient_to_nearest_strand,
    )

    if max_workers in {0, 1}:
        processed: Any = map(worker, sequence_pairs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=max_workers)
        processed = executor.map(
            worker,
            sequence_pairs,
            chunksize=chunksize,
        )

    try:
        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                pass
            else:
                processed = tqdm(
                    processed,
                    total=len(sequence_pairs),
                    desc="Adding same-strand transcript TSSs",
                )
        return list(processed)
    finally:
        if executor is not None:
            executor.shutdown()
