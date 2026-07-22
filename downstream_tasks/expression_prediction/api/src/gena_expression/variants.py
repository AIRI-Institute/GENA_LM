"""Variant parsing and sequence-pair materialization."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from .sequences import AnnotatedSequence, Feature, SequencePair


@dataclass(frozen=True)
class Variant:
    """Substitution, insertion, deletion, or allele replacement.

    ``pos`` is a 0-based reference start. Use ``None`` when the variant is only
    defined by a reference/alternative sequence pair.
    """

    chrom: str | None
    pos: int | None
    ref: str
    alt: str
    id: str | None = None
    strand: str = "+"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Normalize alleles and validate the 0-based variant position."""

        ref = "" if self.ref == "-" else self.ref.upper()
        alt = "" if self.alt == "-" else self.alt.upper()
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "alt", alt)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if self.pos is not None and self.pos < 0:
            raise ValueError("Variant position must be 0-based and non-negative.")

    @classmethod
    def from_str(
        cls,
        text: str,
        *,
        genome: Any | None = None,
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
    ) -> "Variant":
        """Parse common variant strings such as ``chr1:12345:A:T`` or ``A>T``.

        A ``-`` allele is interpreted as an empty allele, matching common
        compact MPRA-table notation for insertions and deletions.
        """

        normalized = re.sub(r"\s+", "", text)
        normalized = normalized.replace(">", ":")
        parts = normalized.split(":")
        if len(parts) != 4:
            raise ValueError(
                "Expected variant string like 'chr1:12345:A:T' or 'chr1:12345:A>T'."
            )
        chrom, pos_text, ref, alt = parts
        if "-" in pos_text:
            pos_text = pos_text.split("-", 1)[0]
        pos = int(pos_text)
        if coordinate_system in {"auto", "1-based"}:
            pos -= 1
        variant = cls(chrom=chrom, pos=pos, ref=ref, alt=alt, id=text)
        if genome is not None:
            variant.validate(genome)
        return variant

    @classmethod
    def from_vcf_record(cls, record: Any, *, genome: Any | None = None) -> "Variant":
        """Create a variant from a VCF-like record object."""

        chrom = getattr(record, "chrom", getattr(record, "CHROM", None))
        pos = int(getattr(record, "pos", getattr(record, "POS"))) - 1
        ref = getattr(record, "ref", getattr(record, "REF"))
        alts = getattr(record, "alts", getattr(record, "ALT", None))
        alt = alts[0] if isinstance(alts, (list, tuple)) else alts
        identifier = getattr(record, "id", getattr(record, "ID", None))
        variant = cls(chrom=chrom, pos=pos, ref=ref, alt=alt, id=identifier)
        if genome is not None:
            variant.validate(genome)
        return variant

    @classmethod
    def from_sequences(
        cls,
        ref: str | AnnotatedSequence,
        alt: str | AnnotatedSequence,
        *,
        name: str | None = None,
    ) -> "Variant":
        """Infer a sequence-only variant from reference and alternative strings.

        If annotated sequences are provided, their features are stored in
        metadata so downstream context builders can preserve annotations such as
        ``variant`` through plasmid or genomic wrapping.
        """

        ref_seq = str(ref).upper()
        alt_seq = str(alt).upper()
        min_len = min(len(ref_seq), len(alt_seq))
        left = 0
        while left < min_len and ref_seq[left] == alt_seq[left]:
            left += 1
        right_ref = len(ref_seq)
        right_alt = len(alt_seq)
        while right_ref > left and right_alt > left and ref_seq[right_ref - 1] == alt_seq[right_alt - 1]:
            right_ref -= 1
            right_alt -= 1
        metadata: dict[str, Any] = {
            "ref_sequence": ref_seq,
            "alt_sequence": alt_seq,
            "ref_change_start": left,
            "ref_change_end": right_ref,
            "alt_change_start": left,
            "alt_change_end": right_alt,
        }
        if isinstance(ref, AnnotatedSequence):
            metadata["ref_name"] = ref.name
            metadata["ref_features"] = [feature.to_dict() for feature in ref.features]
        if isinstance(alt, AnnotatedSequence):
            metadata["alt_name"] = alt.name
            metadata["alt_features"] = [feature.to_dict() for feature in alt.features]

        return cls(
            chrom=None,
            pos=left,
            ref=ref_seq[left:right_ref],
            alt=alt_seq[left:right_alt],
            id=name,
            metadata=metadata,
        )

    def validate(self, genome: Any) -> None:
        """Warn if the reference allele does not match a genome object."""

        if self.chrom is None or self.pos is None:
            raise ValueError("Cannot validate a sequence-only variant without chrom and pos.")
        observed = genome.fetch(self.chrom, self.pos, self.pos + len(self.ref))
        if observed.upper() != self.ref.upper():
            warnings.warn(
                "Reference allele mismatch during variant validation. "
                + self._debug_context(observed=observed, local_start=0),
                RuntimeWarning,
                stacklevel=2,
            )

    def to_sequence_pair(
        self,
        context: Any | None = None,
        *,
        genome: Any | None = None,
        name: str | None = None,
        window_bp: int | None = None,
        primer_pairs: Mapping[str, tuple[str, str]] | Iterable[tuple[str, str]] | None = None,
        primer_locations: Mapping[str, tuple[str, int, int]] | None = None,
        primer_window_bp: int | None = None,
        primer_flank_bp: int = 100,
        primer_location_padding_bp: int | Mapping[str, int] = 0,
        primer_min_match: int = 12,
        primer_edge_slop: int = 25,
        include_primer_tails: bool = False,
        primer_pair_name: str | None = None,
    ) -> SequencePair:
        """Materialize this variant as a reference/alternative sequence pair.

        There are three supported modes:

        1. If ``primer_pairs`` is provided, fetch ``primer_window_bp`` bases
           around the variant, find the matching primer pair, and return the
           MPRA fragment in primer/construct orientation. In this mode
           ``context`` is intentionally ignored because the primers define the
           fragment boundaries.
        2. If ``window_bp`` is provided, fetch a variant-centered genomic
           sequence of that length.
        3. Otherwise, delegate to the provided ``context`` object, preserving
           the original API behavior.

        Set ``include_primer_tails=True`` to return
        ``left_tail + mpra_fragment + right_tail``. The right tail is the
        reverse-complement of the reverse-primer cloning tail, matching the
        plasmid insertion helpers.
        """

        if primer_pairs is not None:
            return self._to_primer_sequence_pair(
                genome=genome,
                primer_pairs=primer_pairs,
                primer_locations=primer_locations,
                primer_window_bp=primer_window_bp,
                primer_flank_bp=primer_flank_bp,
                primer_location_padding_bp=primer_location_padding_bp,
                min_match=primer_min_match,
                edge_slop=primer_edge_slop,
                include_primer_tails=include_primer_tails,
                primer_pair_name=primer_pair_name,
                name=name,
            )

        if window_bp is not None:
            from .contexts import GenomeContext

            return GenomeContext(length=window_bp).build(self, genome=genome, name=name)

        if context is None:
            raise ValueError("Provide context=..., window_bp=..., or primer_pairs=....")
        return context.build(self, genome=genome, name=name)

    def _to_primer_sequence_pair(
        self,
        *,
        genome: Any | None,
        primer_pairs: Mapping[str, tuple[str, str]] | Iterable[tuple[str, str]],
        primer_locations: Mapping[str, tuple[str, int, int]] | None,
        primer_window_bp: int | None,
        primer_flank_bp: int,
        primer_location_padding_bp: int | Mapping[str, int],
        min_match: int,
        edge_slop: int,
        include_primer_tails: bool,
        primer_pair_name: str | None,
        name: str | None,
    ) -> SequencePair:
        """Build a sequence pair by matching cloning primers around the variant."""

        from .contexts.plasmid import reverse_complement

        if genome is None:
            raise ValueError("Primer-based to_sequence_pair() requires genome=....")
        if self.chrom is None or self.pos is None:
            raise ValueError("Primer-based to_sequence_pair() requires chrom and pos.")

        search_specs = self._primer_search_specs(
            genome=genome,
            primer_pairs=primer_pairs,
            primer_locations=primer_locations,
            primer_window_bp=primer_window_bp,
            primer_flank_bp=primer_flank_bp,
            primer_location_padding_bp=primer_location_padding_bp,
            primer_pair_name=primer_pair_name,
        )

        errors: list[str] = []
        selected: tuple[str, str, str, dict[str, object], str, int] | None = None
        for candidate_pairs, start, ref_search, local_variant_start, forced_bounds in search_specs:
            ref_mismatch: dict[str, object] | None = None
            if self.ref and ref_search[local_variant_start : local_variant_start + len(self.ref)] != self.ref:
                observed = ref_search[local_variant_start : local_variant_start + len(self.ref)]
                ref_mismatch = {
                    "expected": self.ref,
                    "observed": observed,
                    "search_start": start,
                    "local_variant_start": local_variant_start,
                    "search_length": len(ref_search),
                }
                warnings.warn(
                    "Reference allele mismatch in primer search window; "
                    "continuing with requested ref/alt replacement. "
                    + self._debug_context(
                        observed=observed,
                        search_start=start,
                        local_start=local_variant_start,
                        search_length=len(ref_search),
                        forced_bounds=forced_bounds,
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            try:
                if forced_bounds is None:
                    pair_name, forward_primer, reverse_primer, match = self._find_primer_match(
                        candidate_pairs,
                        ref_search,
                        min_match=min_match,
                        edge_slop=edge_slop,
                        primer_pair_name=primer_pair_name,
                        local_variant_start=local_variant_start,
                        ref_len=len(self.ref),
                    )
                else:
                    pair_name, forward_primer, reverse_primer, match = self._find_location_primer_match(
                        candidate_pairs,
                        ref_search,
                        expected_position=forced_bounds[0],
                        min_match=min_match,
                        edge_slop=edge_slop,
                        primer_pair_name=primer_pair_name,
                    )
                    match["forced_fragment_bounds"] = forced_bounds
                if ref_mismatch is not None:
                    match["reference_mismatch"] = ref_mismatch
            except ValueError as exc:
                errors.append(
                    f"{exc} | "
                    + self._debug_context(
                        search_start=start,
                        local_start=local_variant_start,
                        search_length=len(ref_search),
                        forced_bounds=forced_bounds,
                    )
                )
                continue
            selected = (pair_name, forward_primer, reverse_primer, match, ref_search, local_variant_start)
            break

        if selected is None:
            detail = "; ".join(errors[:5])
            raise ValueError(
                "Could not infer primer-defined fragment for variant. "
                + self._debug_context(
                    primer_locations=primer_locations is not None,
                    primer_window_bp=primer_window_bp,
                    primer_flank_bp=primer_flank_bp,
                    primer_location_padding_bp=primer_location_padding_bp,
                    primer_pair_name=primer_pair_name,
                    candidate_windows=len(search_specs),
                )
                + f" Candidate errors: {detail}"
            )

        pair_name, forward_primer, reverse_primer, match, ref_search, local_variant_start = selected
        orientation = str(match["orientation"])

        # Match positions are reported on the oriented sequence used by the
        # forward primer and the reverse-complement target used by the reverse
        # primer. This converts them into an oriented MPRA-fragment slice.
        oriented_ref_search = ref_search if orientation == "plus" else reverse_complement(ref_search)
        alt_search = (
            ref_search[:local_variant_start]
            + self.alt
            + ref_search[local_variant_start + len(self.ref) :]
        )
        oriented_alt_search = alt_search if orientation == "plus" else reverse_complement(alt_search)
        fragment_start, fragment_end, variant_start = self._primer_fragment_bounds(
            match,
            sequence_length=len(ref_search),
            local_variant_start=local_variant_start,
            ref_len=len(self.ref),
        )
        if fragment_start >= fragment_end:
            raise ValueError(
                f"Primer pair {pair_name!r} produced invalid fragment bounds "
                f"{fragment_start}:{fragment_end}. "
                + self._debug_context(
                    pair_name=pair_name,
                    orientation=orientation,
                    search_length=len(ref_search),
                    local_start=local_variant_start,
                    match=dict(match),
                )
            )

        ref_fragment = oriented_ref_search[fragment_start:fragment_end]
        alt_fragment = oriented_alt_search[fragment_start : fragment_end + self.length_change()]
        variant_strand = "+"
        if orientation == "minus":
            variant_strand = "-"

        if not (0 <= variant_start <= len(ref_fragment)):
            raise ValueError(
                f"Variant falls outside fragment inferred from primer pair {pair_name!r}. "
                + self._debug_context(
                    pair_name=pair_name,
                    orientation=orientation,
                    fragment_start=fragment_start,
                    fragment_end=fragment_end,
                    fragment_length=len(ref_fragment),
                    variant_start=variant_start,
                    local_start=local_variant_start,
                    search_length=len(ref_search),
                    match=dict(match),
                )
            )

        left_tail = str(match["forward_tail"])
        reverse_primer_tail = str(match["reverse_tail"])
        right_tail = reverse_complement(reverse_primer_tail)
        ref_seq, alt_seq, variant_offset = self._annotate_primer_fragment(
            ref_fragment=ref_fragment,
            alt_fragment=alt_fragment,
            variant_start=variant_start,
            include_primer_tails=include_primer_tails,
            left_tail=left_tail,
            right_tail=right_tail,
            name=name or self.id or pair_name,
            pair_name=pair_name,
            orientation=orientation,
            variant_strand=variant_strand,
            match=match,
        )
        pair = SequencePair(
            ref=ref_seq,
            alt=alt_seq,
            variant=self,
            metadata={
                "context": "primer_pair",
                "primer_pair": pair_name,
                "orientation": orientation,
                "forward_primer": forward_primer,
                "reverse_primer": reverse_primer,
                "primer_window_bp": primer_window_bp,
                "include_primer_tails": include_primer_tails,
                "variant_offset": variant_offset,
                "match": dict(match),
            },
        )
        pair.assert_compatible()
        return pair

    def _debug_context(self, **extra: Any) -> str:
        """Return compact variant diagnostics for warnings and exceptions."""

        data = {
            "variant_id": self.id,
            "chrom": self.chrom,
            "pos0": self.pos,
            "pos1": None if self.pos is None else self.pos + 1,
            "ref": self.ref,
            "alt": self.alt,
            "ref_len": len(self.ref),
            "alt_len": len(self.alt),
        }
        data.update(extra)
        return "debug=" + repr(data)

    def _primer_search_specs(
        self,
        *,
        genome: Any,
        primer_pairs: Mapping[str, tuple[str, str]] | Iterable[tuple[str, str]],
        primer_locations: Mapping[str, tuple[str, int, int]] | None,
        primer_window_bp: int | None,
        primer_flank_bp: int,
        primer_location_padding_bp: int | Mapping[str, int],
        primer_pair_name: str | None,
    ) -> list[tuple[Mapping[str, tuple[str, str]] | Iterable[tuple[str, str]], int, str, int, tuple[int, int] | None]]:
        """Build genomic search windows for primer-defined MPRA fragments.

        When known primer locations are available, this mirrors the older
        Table-18 workflow: choose intervals whose coordinates contain the
        variant and fetch the whole construct plus flank. Otherwise it falls
        back to a variant-centered search window.
        """

        if primer_locations is None:
            if primer_window_bp is None:
                raise ValueError("Primer-based to_sequence_pair() requires primer_window_bp=....")
            if primer_window_bp <= 0:
                raise ValueError("primer_window_bp must be positive.")
            start = self.pos - primer_window_bp // 2  # type: ignore[operator]
            end = start + primer_window_bp
            ref_search = genome.fetch(self.chrom, start, end).upper()
            return [(primer_pairs, start, ref_search, self.pos - start, None)]  # type: ignore[operator]

        if not isinstance(primer_pairs, Mapping):
            raise TypeError("primer_locations requires primer_pairs to be a mapping with matching keys.")
        if primer_flank_bp < 0:
            raise ValueError("primer_flank_bp must be non-negative.")

        variant_start = int(self.pos)  # 0-based
        variant_end = variant_start + max(1, len(self.ref))
        query_chrom = self._normalize_chrom_name(str(self.chrom))
        specs: list[tuple[Mapping[str, tuple[str, str]], int, str, int]] = []

        for key, (chrom, start_1based, end_1based) in primer_locations.items():
            if primer_pair_name is not None and key != primer_pair_name:
                continue
            if key not in primer_pairs:
                continue
            if self._normalize_chrom_name(chrom) != query_chrom:
                continue

            padding = self._primer_location_padding_for(key, primer_location_padding_bp)
            interval_start = int(start_1based) - 1 - padding
            interval_end = int(end_1based) + padding
            if not (interval_start <= variant_start and variant_end <= interval_end):
                continue

            fetch_start = interval_start - primer_flank_bp
            fetch_end = interval_end + primer_flank_bp
            ref_search = genome.fetch(chrom, fetch_start, fetch_end).upper()
            forced_bounds = (primer_flank_bp, primer_flank_bp + (interval_end - interval_start))
            specs.append(({key: primer_pairs[key]}, fetch_start, ref_search, variant_start - fetch_start, forced_bounds))

        if not specs:
            raise ValueError(
                "No primer location interval contained the variant. Check "
                "coordinate_system, chromosome names, or primer_locations. "
                + self._debug_context(
                    query_chrom=query_chrom,
                    variant_start0=variant_start,
                    variant_end0=variant_end,
                    location_count=len(primer_locations),
                    primer_location_padding_bp=primer_location_padding_bp,
                    primer_pair_name=primer_pair_name,
                )
            )
        return specs

    @staticmethod
    def _primer_location_padding_for(key: str, padding: int | Mapping[str, int]) -> int:
        """Return coordinate padding for one primer-location interval."""

        if isinstance(padding, Mapping):
            value = int(padding.get(key, 0))
        else:
            value = int(padding)
        if value < 0:
            raise ValueError("primer_location_padding_bp must be non-negative.")
        return value

    @staticmethod
    def _normalize_chrom_name(chrom: str) -> str:
        """Normalize chromosome names for comparing table and FASTA labels."""

        normalized = chrom.lower()
        if normalized.startswith("chr"):
            normalized = normalized[3:]
        if normalized == "m":
            return "mt"
        return normalized

    @staticmethod
    def _find_location_primer_match(
        primer_pairs: Mapping[str, tuple[str, str]] | Iterable[tuple[str, str]],
        sequence: str,
        *,
        expected_position: int,
        min_match: int,
        edge_slop: int,
        primer_pair_name: str | None,
    ) -> tuple[str, str, str, dict[str, object]]:
        """Match primers in a known interval-backed search window."""

        from .contexts.plasmid import _match_primer_pair

        if isinstance(primer_pairs, Mapping):
            items = list(primer_pairs.items())
        else:
            items = [(f"primer_pair_{idx + 1}", pair) for idx, pair in enumerate(primer_pairs)]

        if primer_pair_name is not None:
            items = [(key, pair) for key, pair in items if key == primer_pair_name]
            if not items:
                raise KeyError(f"Primer pair {primer_pair_name!r} was not found.")

        matches: list[tuple[str, str, str, dict[str, object]]] = []
        for key, pair in items:
            forward_primer, reverse_primer = pair
            try:
                match = _match_primer_pair(
                    forward_primer,
                    reverse_primer,
                    sequence,
                    min_match=min_match,
                    expected_position=expected_position,
                    edge_slop=edge_slop,
                )
            except ValueError:
                continue
            matches.append((str(key), forward_primer, reverse_primer, match))

        if not matches:
            raise ValueError(
                "No primer pair matched the known primer-location interval. "
                f"debug={{'expected_position': {expected_position!r}, "
                f"'sequence_length': {len(sequence)!r}, "
                f"'min_match': {min_match!r}, "
                f"'edge_slop': {edge_slop!r}, "
                f"'primer_pair_name': {primer_pair_name!r}}}"
            )

        return max(
            matches,
            key=lambda item: (
                -int(item[3]["edge_distance"]),
                min(int(item[3]["forward_match_len"]), int(item[3]["reverse_match_len"])),
                int(item[3]["forward_match_len"]) + int(item[3]["reverse_match_len"]),
                str(item[0]),
            ),
        )

    @staticmethod
    def _find_primer_match(
        primer_pairs: Mapping[str, tuple[str, str]] | Iterable[tuple[str, str]],
        sequence: str,
        *,
        min_match: int,
        edge_slop: int,
        primer_pair_name: str | None,
        local_variant_start: int,
        ref_len: int,
    ) -> tuple[str, str, str, dict[str, object]]:
        """Return the best primer pair whose inferred fragment contains the variant."""

        from .contexts.plasmid import reverse_complement

        if isinstance(primer_pairs, Mapping):
            items = list(primer_pairs.items())
        else:
            items = [(f"primer_pair_{idx + 1}", pair) for idx, pair in enumerate(primer_pairs)]

        if primer_pair_name is not None:
            items = [(key, pair) for key, pair in items if key == primer_pair_name]
            if not items:
                raise KeyError(f"Primer pair {primer_pair_name!r} was not found.")

        matches: list[tuple[str, str, str, dict[str, object]]] = []
        for key, pair in items:
            forward_primer, reverse_primer = pair
            orientations = {
                "plus": (sequence, reverse_complement(sequence)),
                "minus": (reverse_complement(sequence), sequence),
            }
            for orientation, (forward_target, reverse_target) in orientations.items():
                forward_candidates = Variant._primer_suffix_candidates(
                    forward_primer,
                    forward_target,
                    min_match=min_match,
                )
                reverse_candidates = Variant._primer_suffix_candidates(
                    reverse_primer,
                    reverse_target,
                    min_match=min_match,
                )

                for forward_candidate in forward_candidates:
                    for reverse_candidate in reverse_candidates:
                        match = {
                            "orientation": orientation,
                            "forward_tail": forward_candidate["tail"],
                            "reverse_tail": reverse_candidate["tail"],
                            "forward_annealing": forward_candidate["annealing"],
                            "reverse_annealing": reverse_candidate["annealing"],
                            "forward_match_len": forward_candidate["match_len"],
                            "reverse_match_len": reverse_candidate["match_len"],
                            "forward_match_position": forward_candidate["position"],
                            "reverse_match_position": reverse_candidate["position"],
                        }
                        fragment_start, fragment_end, _ = Variant._primer_fragment_bounds(
                            match,
                            sequence_length=len(sequence),
                            local_variant_start=local_variant_start,
                            ref_len=ref_len,
                        )
                        fragment_length = fragment_end - fragment_start
                        if fragment_length <= 0:
                            continue
                        if not Variant._primer_match_contains_variant(
                            match,
                            sequence_length=len(sequence),
                            local_variant_start=local_variant_start,
                            ref_len=ref_len,
                        ):
                            continue

                        # Kept for compatibility with older match diagnostics.
                        # In variant-centered searches this is only a weak tie
                        # breaker; the primer hits are not expected at window edges.
                        match["edge_distance"] = abs(int(forward_candidate["position"])) + abs(
                            int(reverse_candidate["position"])
                        )
                        match["fragment_length"] = fragment_length
                        matches.append((str(key), forward_primer, reverse_primer, match))

        if not matches:
            raise ValueError(
                "No primer pair produced an inferred fragment containing the variant. "
                + Variant(
                    chrom=None,
                    pos=local_variant_start,
                    ref="N" * ref_len,
                    alt="",
                )._debug_context(
                    local_start=local_variant_start,
                    ref_len=ref_len,
                    sequence_length=len(sequence),
                    min_match=min_match,
                    edge_slop=edge_slop,
                    primer_pair_name=primer_pair_name,
                )
            )

        return max(
            matches,
            key=lambda item: (
                min(int(item[3]["forward_match_len"]), int(item[3]["reverse_match_len"])),
                int(item[3]["forward_match_len"]) + int(item[3]["reverse_match_len"]),
                -int(item[3]["fragment_length"]),
                -int(item[3]["edge_distance"]),
                str(item[0]),
            ),
        )

    @staticmethod
    def _primer_suffix_candidates(
        primer: str,
        target: str,
        *,
        min_match: int,
    ) -> list[dict[str, object]]:
        """Return every suffix match that could represent primer annealing."""

        primer = primer.upper()
        target = target.upper()
        candidates: list[dict[str, object]] = []
        for match_len in range(len(primer), min_match - 1, -1):
            annealing = primer[-match_len:]
            start = 0
            while True:
                position = target.find(annealing, start)
                if position == -1:
                    break
                candidates.append(
                    {
                        "tail": primer[:-match_len],
                        "annealing": annealing,
                        "match_len": match_len,
                        "position": position,
                    }
                )
                start = position + 1
        return candidates

    @staticmethod
    def _primer_fragment_bounds(
        match: Mapping[str, object],
        *,
        sequence_length: int,
        local_variant_start: int,
        ref_len: int,
    ) -> tuple[int, int, int]:
        """Return oriented fragment bounds and variant offset for one primer match."""

        orientation = str(match["orientation"])
        if "forced_fragment_bounds" in match:
            plus_start, plus_end = match["forced_fragment_bounds"]  # type: ignore[misc]
            if orientation == "minus":
                fragment_start = sequence_length - int(plus_end)
                fragment_end = sequence_length - int(plus_start)
            else:
                fragment_start = int(plus_start)
                fragment_end = int(plus_end)
        else:
            fragment_start = int(match["forward_match_position"])
            fragment_end = sequence_length - int(match["reverse_match_position"])

        if orientation == "minus":
            variant_start = sequence_length - local_variant_start - ref_len - fragment_start
        else:
            variant_start = local_variant_start - fragment_start
        return fragment_start, fragment_end, variant_start

    @staticmethod
    def _primer_match_contains_variant(
        match: Mapping[str, object],
        *,
        sequence_length: int,
        local_variant_start: int,
        ref_len: int,
    ) -> bool:
        """Return whether a primer match produces a fragment containing the variant."""

        fragment_start, fragment_end, variant_start = Variant._primer_fragment_bounds(
            match,
            sequence_length=sequence_length,
            local_variant_start=local_variant_start,
            ref_len=ref_len,
        )
        fragment_length = fragment_end - fragment_start
        return fragment_length > 0 and 0 <= variant_start <= fragment_length

    def _annotate_primer_fragment(
        self,
        *,
        ref_fragment: str,
        alt_fragment: str,
        variant_start: int,
        include_primer_tails: bool,
        left_tail: str,
        right_tail: str,
        name: str,
        pair_name: str,
        orientation: str,
        variant_strand: str,
        match: Mapping[str, object],
    ) -> tuple[AnnotatedSequence, AnnotatedSequence, int]:
        """Create annotated reference/alternative sequences for a primer hit."""

        fragment_features = (
            Feature(
                "mpra_fragment",
                0,
                len(ref_fragment),
                type="mpra_fragment",
                source="genome",
                metadata={"primer_pair": pair_name, "orientation": orientation, "match": dict(match)},
            ),
            Feature(
                "variant",
                variant_start,
                variant_start + max(1, len(self.ref)),
                type="variant",
                strand=variant_strand,
                source="variant",
                metadata=self.to_dict(),
            ),
        )
        ref = AnnotatedSequence(ref_fragment, name=f"{name}_ref", features=fragment_features)
        alt = AnnotatedSequence(
            alt_fragment,
            name=f"{name}_alt",
            features=(
                fragment_features[0],
                Feature(
                    "variant",
                    variant_start,
                    variant_start + max(1, len(self.alt)),
                    type="variant",
                    strand=variant_strand,
                    source="variant",
                    metadata=self.to_dict(),
                ),
            ),
        )
        if not include_primer_tails:
            return ref, alt, variant_start

        left = AnnotatedSequence(left_tail, name="left_primer_tail").add_feature(
            "left_mcs_scar", 0, len(left_tail), type="scar", source="primer"
        )
        right = AnnotatedSequence(right_tail, name="right_primer_tail").add_feature(
            "right_mcs_scar", 0, len(right_tail), type="scar", source="primer"
        )
        return (
            AnnotatedSequence.concat(left, ref, right, name=f"{name}_ref"),
            AnnotatedSequence.concat(left, alt, right, name=f"{name}_alt"),
            variant_start + len(left_tail),
        )

    def as_feature(self, name: str = "variant") -> Feature:
        """Return this variant as a sequence feature."""

        if self.pos is None:
            start = 0
        else:
            start = self.pos
        end = start + max(1, len(self.ref))
        return Feature(
            name=name,
            start=start,
            end=end,
            type="variant",
            strand=self.strand,
            source="variant",
            metadata=self.to_dict(),
        )

    def length_change(self) -> int:
        """Return ``len(alt) - len(ref)``."""

        return len(self.alt) - len(self.ref)

    def is_snv(self) -> bool:
        """Return ``True`` for a single-nucleotide substitution."""

        return len(self.ref) == len(self.alt) == 1 and self.ref != self.alt

    def is_indel(self) -> bool:
        """Return ``True`` when the variant changes sequence length."""

        return len(self.ref) != len(self.alt)

    def apply_to(self, sequence: AnnotatedSequence, *, offset: int = 0) -> AnnotatedSequence:
        """Apply this variant to an annotated sequence.

        ``offset`` is the sequence coordinate corresponding to genomic
        ``self.pos``. For a window built around the variant this is usually the
        variant's local position in that window.
        """

        start = offset
        end = offset + len(self.ref)
        observed = sequence.sequence[start:end]
        if self.ref and observed.upper() != self.ref.upper():
            warnings.warn(
                "Reference allele mismatch in annotated sequence context; "
                "continuing with requested ref/alt replacement. "
                + self._debug_context(
                    observed=observed,
                    sequence_name=sequence.name,
                    sequence_length=len(sequence),
                    local_start=start,
                    local_end=end,
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        return sequence.replace(start, end, self.alt, preserve_partial_features=True).add_feature(
            "variant",
            start,
            start + max(1, len(self.alt)),
            type="variant",
            source="variant",
            metadata=self.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this variant."""

        return {
            "chrom": self.chrom,
            "pos": self.pos,
            "ref": self.ref,
            "alt": self.alt,
            "id": self.id,
            "strand": self.strand,
            "metadata": dict(self.metadata or {}),
        }

    def plot(
        self,
        *,
        ref_sequence: str | AnnotatedSequence | None = None,
        alt_sequence: str | AnnotatedSequence | None = None,
        flank: int = 40,
        ax: Any | None = None,
        figsize: tuple[float, float] = (12.0, 2.4),
        save_path: str | Path | None = None,
    ):
        """Plot where the variant changes sequence.

        If full reference and alternative context sequences are provided, the
        plot shows the local changed segment in those contexts. Otherwise it
        shows the allele-level ``ref -> alt`` change.
        """

        import matplotlib.pyplot as plt

        if ref_sequence is not None and alt_sequence is not None:
            pair = SequencePair(
                ref=ref_sequence if isinstance(ref_sequence, AnnotatedSequence) else AnnotatedSequence(str(ref_sequence)),
                alt=alt_sequence if isinstance(alt_sequence, AnnotatedSequence) else AnnotatedSequence(str(alt_sequence)),
                variant=self,
            )
            return pair.plot_difference(flank=flank, ax=ax, figsize=figsize, save_path=save_path)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        fig = ax.figure
        ax.axis("off")
        label = self.id or (
            f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"
            if self.chrom is not None and self.pos is not None
            else f"{self.ref}>{self.alt}"
        )
        ax.text(0.02, 0.72, "REF", weight="bold", transform=ax.transAxes)
        ax.text(0.12, 0.72, self.ref or "-", family="monospace", transform=ax.transAxes)
        ax.text(0.02, 0.38, "ALT", weight="bold", transform=ax.transAxes)
        ax.text(0.12, 0.38, self.alt or "-", family="monospace", transform=ax.transAxes)
        ax.text(0.02, 0.08, label, fontsize=9, transform=ax.transAxes)
        ax.set_title("Variant allele change")
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig, ax
