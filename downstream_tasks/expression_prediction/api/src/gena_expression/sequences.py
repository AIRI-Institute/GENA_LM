"""Annotated DNA sequence objects and coordinate bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence


_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass(frozen=True)
class Feature:
    """One interval annotation on an :class:`AnnotatedSequence`.

    Coordinates are 0-based, half-open, and relative to the sequence that owns
    the feature.
    """

    name: str
    start: int
    end: int
    type: str = "feature"
    strand: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid feature interval: {self.start}:{self.end}")

    def length(self) -> int:
        """Return feature length in bases."""

        return self.end - self.start

    def copy(self, **changes: Any) -> "Feature":
        """Return a copy with selected fields changed."""

        return replace(self, **changes)

    def shift(self, offset: int) -> "Feature":
        """Return the same feature shifted by ``offset`` bases."""

        return replace(self, start=self.start + offset, end=self.end + offset)

    def clip(self, start: int, end: int) -> "Feature | None":
        """Clip this feature to ``[start, end)`` and return sequence-local coordinates."""

        clipped_start = max(self.start, start)
        clipped_end = min(self.end, end)
        if clipped_end <= clipped_start:
            return None
        return replace(self, start=clipped_start - start, end=clipped_end - start)

    def overlaps(self, start: int, end: int) -> bool:
        """Return ``True`` when this feature overlaps ``[start, end)``."""

        return self.start < end and start < self.end

    def contains(self, position: int) -> bool:
        """Return ``True`` when ``position`` lies inside the feature."""

        return self.start <= position < self.end

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feature to a JSON-friendly dictionary."""

        return {
            "name": self.name,
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "strand": self.strand,
            "source": self.source,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class SourceCoordinate:
    """A source coordinate corresponding to one sequence position."""

    source: str
    position: int | None
    chrom: str | None = None
    strand: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CoordinateSegment:
    """A contiguous mapping from sequence span to one biological source."""

    seq_start: int
    seq_end: int
    source: str
    chrom: str | None = None
    source_start: int | None = None
    source_end: int | None = None
    strand: str | None = None
    metadata: Mapping[str, Any] | None = None

    def shifted(self, offset: int) -> "CoordinateSegment":
        """Return the same segment with sequence coordinates shifted."""

        return replace(self, seq_start=self.seq_start + offset, seq_end=self.seq_end + offset)


@dataclass(frozen=True)
class CoordinateMap:
    """Map positions in an annotated sequence back to source coordinates."""

    segments: Sequence[CoordinateSegment] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))

    @classmethod
    def from_length(cls, length: int, *, source: str = "sequence") -> "CoordinateMap":
        """Create a trivial map covering a sequence of ``length`` bases."""

        if length <= 0:
            return cls(())
        return cls((CoordinateSegment(0, length, source=source, source_start=0, source_end=length),))

    def seq_to_source(self, position: int) -> SourceCoordinate | None:
        """Map one sequence coordinate to its source coordinate, when known."""

        for segment in self.segments:
            if segment.seq_start <= position < segment.seq_end:
                source_position: int | None = None
                if segment.source_start is not None and segment.source_end is not None:
                    offset = position - segment.seq_start
                    if segment.strand == "-":
                        source_position = segment.source_end - offset - 1
                    else:
                        source_position = segment.source_start + offset
                return SourceCoordinate(
                    source=segment.source,
                    position=source_position,
                    chrom=segment.chrom,
                    strand=segment.strand,
                    metadata=segment.metadata,
                )
        return None

    def source_to_seq(
        self,
        source: str,
        position: int,
        chrom: str | None = None,
    ) -> int | None:
        """Map a source coordinate back to sequence coordinate, when possible."""

        for segment in self.segments:
            if segment.source != source:
                continue
            if chrom is not None and segment.chrom != chrom:
                continue
            if segment.source_start is None or segment.source_end is None:
                continue
            if not (segment.source_start <= position < segment.source_end):
                continue
            if segment.strand == "-":
                return segment.seq_start + (segment.source_end - position - 1)
            return segment.seq_start + (position - segment.source_start)
        return None

    def slice(self, start: int, end: int) -> "CoordinateMap":
        """Return a coordinate map clipped to ``[start, end)`` and shifted to zero."""

        clipped: list[CoordinateSegment] = []
        for segment in self.segments:
            overlap_start = max(start, segment.seq_start)
            overlap_end = min(end, segment.seq_end)
            if overlap_end <= overlap_start:
                continue
            source_start = segment.source_start
            source_end = segment.source_end
            if source_start is not None and source_end is not None:
                left_delta = overlap_start - segment.seq_start
                right_delta = segment.seq_end - overlap_end
                if segment.strand == "-":
                    source_start = source_start + right_delta
                    source_end = source_end - left_delta
                else:
                    source_start = source_start + left_delta
                    source_end = source_end - right_delta
            clipped.append(
                CoordinateSegment(
                    seq_start=overlap_start - start,
                    seq_end=overlap_end - start,
                    source=segment.source,
                    chrom=segment.chrom,
                    source_start=source_start,
                    source_end=source_end,
                    strand=segment.strand,
                    metadata=segment.metadata,
                )
            )
        return CoordinateMap(clipped)

    def reverse_complement(self, sequence_length: int) -> "CoordinateMap":
        """Return a map for the reverse-complemented sequence."""

        reversed_segments = []
        for segment in reversed(self.segments):
            strand = {"+": "-", "-": "+"}.get(segment.strand, segment.strand)
            reversed_segments.append(
                CoordinateSegment(
                    seq_start=sequence_length - segment.seq_end,
                    seq_end=sequence_length - segment.seq_start,
                    source=segment.source,
                    chrom=segment.chrom,
                    source_start=segment.source_start,
                    source_end=segment.source_end,
                    strand=strand,
                    metadata=segment.metadata,
                )
            )
        return CoordinateMap(reversed_segments)

    def concat(self, other: "CoordinateMap", offset: int) -> "CoordinateMap":
        """Concatenate another map, shifting its sequence coordinates by ``offset``."""

        return CoordinateMap(tuple(self.segments) + tuple(seg.shifted(offset) for seg in other.segments))

    def to_frame(self):
        """Return the coordinate map as a pandas DataFrame."""

        import pandas as pd

        return pd.DataFrame([segment.__dict__ for segment in self.segments])


@dataclass(frozen=True)
class AnnotatedSequence:
    """DNA sequence string plus annotations, coordinate map, and provenance."""

    sequence: str
    name: str | None = None
    features: Sequence[Feature] = field(default_factory=tuple)
    coordinate_map: CoordinateMap | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        sequence = self.sequence.upper()
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "features", tuple(self.features))
        if self.coordinate_map is None:
            object.__setattr__(self, "coordinate_map", CoordinateMap.from_length(len(sequence)))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def __str__(self) -> str:
        return self.sequence

    def __len__(self) -> int:
        return len(self.sequence)

    def __getitem__(self, item: slice | int) -> str | "AnnotatedSequence":
        if isinstance(item, int):
            return self.sequence[item]
        start, stop, step = item.indices(len(self))
        if step != 1:
            return self.sequence[item]
        return AnnotatedSequence(
            self.sequence[start:stop],
            name=self.name,
            features=tuple(feature for feature in (f.clip(start, stop) for f in self.features) if feature),
            coordinate_map=self.coordinate_map.slice(start, stop),
            metadata=self.metadata,
        )

    def add_feature(
        self,
        name: str,
        start: int,
        end: int,
        type: str = "feature",
        strand: str | None = None,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AnnotatedSequence":
        """Return a copy with one additional feature."""

        return replace(
            self,
            features=tuple(self.features)
            + (Feature(name, start, end, type=type, strand=strand, source=source, metadata=metadata),),
        )

    def feature(self, name: str, *, required: bool = True) -> Feature | None:
        """Return the first feature with ``name``."""

        for feature in self.features:
            if feature.name == name or feature.type == name:
                return feature
        if required:
            raise KeyError(f"Feature {name!r} was not found on sequence {self.name!r}.")
        return None

    def features_at(self, position: int) -> list[Feature]:
        """Return all features containing ``position``."""

        return [feature for feature in self.features if feature.contains(position)]

    def features_overlapping(self, start: int, end: int) -> list[Feature]:
        """Return all features overlapping ``[start, end)``."""

        return [feature for feature in self.features if feature.overlaps(start, end)]

    def update_feature(self, feature: str | Feature, **changes: Any) -> "AnnotatedSequence":
        """Return a copy with one resolved feature changed.

        String queries use the same unique substring matching as ``select()``.
        Use ``update_features()`` when multiple features should be changed.
        """

        feature_index = self._resolve_unique_feature_index(feature)
        features = tuple(
            item.copy(**changes) if index == feature_index else item
            for index, item in enumerate(self.features)
        )
        return replace(self, features=features)

    def update_features(
        self,
        where: Callable[[Feature], bool],
        **changes: Any,
    ) -> "AnnotatedSequence":
        """Return a copy with every matching feature changed."""

        features = tuple(item.copy(**changes) if where(item) else item for item in self.features)
        return replace(self, features=features)

    def remove_feature(self, feature: str | Feature) -> "AnnotatedSequence":
        """Return a copy with one resolved feature removed."""

        feature_index = self._resolve_unique_feature_index(feature)
        features = tuple(item for index, item in enumerate(self.features) if index != feature_index)
        return replace(self, features=features)

    def remove_features(
        self,
        where: Callable[[Feature], bool] | None = None,
        *,
        name: str | None = None,
        type: str | None = None,
        source: str | None = None,
    ) -> "AnnotatedSequence":
        """Return a copy with every matching feature removed."""

        self._require_feature_filter(where, name=name, type=type, source=source)
        features = tuple(
            item
            for item in self.features
            if not self._matches_feature_filter(
                item,
                where,
                name=name,
                type=type,
                source=source,
            )
        )
        return replace(self, features=features)

    def keep_features(
        self,
        where: Callable[[Feature], bool] | None = None,
        *,
        name: str | None = None,
        type: str | None = None,
        source: str | None = None,
    ) -> "AnnotatedSequence":
        """Return a copy containing only matching features."""

        self._require_feature_filter(where, name=name, type=type, source=source)
        features = tuple(
            item
            for item in self.features
            if self._matches_feature_filter(
                item,
                where,
                name=name,
                type=type,
                source=source,
            )
        )
        return replace(self, features=features)

    def map_features(self, fn: Callable[[Feature], Feature | None]) -> "AnnotatedSequence":
        """Return a copy after applying ``fn`` to every feature.

        The callable may return a changed :class:`Feature`, the original
        feature, or ``None`` to drop that annotation.
        """

        features: list[Feature] = []
        for feature in self.features:
            mapped = fn(feature)
            if mapped is None:
                continue
            if not isinstance(mapped, Feature):
                raise TypeError("map_features() callable must return Feature or None.")
            features.append(mapped)
        return replace(self, features=tuple(features))

    @staticmethod
    def _require_feature_filter(
        where: Callable[[Feature], bool] | None,
        *,
        name: str | None,
        type: str | None,
        source: str | None,
    ) -> None:
        """Require at least one feature filter for bulk keep/remove calls."""

        if where is None and name is None and type is None and source is None:
            raise ValueError("Provide where, name, type, or source.")

    @staticmethod
    def _matches_feature_filter(
        feature: Feature,
        where: Callable[[Feature], bool] | None = None,
        *,
        name: str | None = None,
        type: str | None = None,
        source: str | None = None,
    ) -> bool:
        """Return whether ``feature`` matches all requested filters."""

        if where is not None and not where(feature):
            return False
        if name is not None and feature.name != name:
            return False
        if type is not None and feature.type != type:
            return False
        if source is not None and feature.source != source:
            return False
        return True

    def select(
        self,
        start: int | str | Feature | Sequence[str | Feature] | None = None,
        end: int | None = None,
        strand: str | None = None,
        *,
        feature: str | Feature | None = None,
        features: Sequence[str | Feature] | None = None,
        name: str | None = None,
    ) -> "AnnotatedSequence":
        """Return an annotated subsequence selected by coordinates or features.

        Supported modes are:

        - ``select(start, end, strand="+")`` for a coordinate interval.
        - ``select(feature="promoter")`` or ``select("promoter")`` for one
          feature. Strings are resolved by substring matching against feature
          name or type; ambiguous matches raise an error.
        - ``select(features=["exon1", "exon2"])`` or ``select([...])`` to
          concatenate feature intervals in the input order.
        """

        requested_strand = self._validate_selection_strand(strand)

        if feature is not None and features is not None:
            raise ValueError("Pass only one of feature= or features=.")

        if feature is None and features is None and start is not None and not isinstance(start, int):
            if end is not None:
                raise ValueError("end can only be used with integer start coordinates.")
            if isinstance(start, (str, Feature)):
                feature = start
            else:
                features = start

        if features is not None:
            feature_items = (features,) if isinstance(features, (str, Feature)) else tuple(features)
            parts = []
            resolved_features = []
            for item in feature_items:
                resolved = self._resolve_unique_feature(item)
                resolved_features.append(resolved)
                part_strand = requested_strand or self._feature_selection_strand(resolved)
                parts.append(self._select_interval(resolved.start, resolved.end, part_strand, name=resolved.name))
            selected = AnnotatedSequence.concat(*parts, name=name or self.name)
            return selected.with_metadata(
                selected_from=self.name,
                selection_mode="features",
                selected_features=[feature.to_dict() for feature in resolved_features],
                selection_strand=requested_strand or "feature",
            )

        if feature is not None:
            resolved = self._resolve_unique_feature(feature)
            selected_strand = requested_strand or self._feature_selection_strand(resolved)
            selected = self._select_interval(
                resolved.start,
                resolved.end,
                selected_strand,
                name=name or resolved.name,
            )
            return selected.with_metadata(
                selected_from=self.name,
                selection_mode="feature",
                selected_feature=resolved.to_dict(),
                selection_strand=selected_strand,
            )

        if not isinstance(start, int) or end is None:
            raise ValueError("Pass start and end coordinates, feature=, or features=.")

        selected_strand = requested_strand or "+"
        return self._select_interval(start, end, selected_strand, name=name or self.name).with_metadata(
            selected_from=self.name,
            selection_mode="coordinates",
            selection_start=start,
            selection_end=end,
            selection_strand=selected_strand,
        )

    def _select_interval(self, start: int, end: int, strand: str, *, name: str | None = None) -> "AnnotatedSequence":
        """Return one validated interval, reverse-complementing when requested."""

        if start < 0 or end < start or end > len(self):
            raise ValueError(f"Invalid selection interval {start}:{end} for sequence length {len(self)}.")
        selected = self[start:end]
        assert isinstance(selected, AnnotatedSequence)
        selected = replace(selected, name=name or selected.name)
        if strand == "-":
            selected = selected.reverse_complement(name=name or selected.name)
        return selected

    def _resolve_unique_feature(self, feature: str | Feature) -> Feature:
        """Resolve a feature object or unique name/type substring."""

        if isinstance(feature, Feature):
            return feature
        matches = [item for item in self.features if feature in item.name or feature in item.type]
        if not matches:
            raise KeyError(f"Feature containing {feature!r} was not found on sequence {self.name!r}.")
        if len(matches) > 1:
            summary = ", ".join(f"{item.name}({item.start}:{item.end}, type={item.type})" for item in matches)
            raise ValueError(f"Feature query {feature!r} matched multiple features: {summary}")
        return matches[0]

    def _resolve_unique_feature_index(self, feature: str | Feature) -> int:
        """Resolve a feature query to exactly one index in ``self.features``."""

        if isinstance(feature, Feature):
            matches = [index for index, item in enumerate(self.features) if item == feature]
            if not matches:
                raise KeyError(f"Feature {feature!r} is not present on sequence {self.name!r}.")
            if len(matches) > 1:
                raise ValueError(f"Feature object matched multiple identical features on sequence {self.name!r}.")
            return matches[0]

        resolved = self._resolve_unique_feature(feature)
        for index, item in enumerate(self.features):
            if item is resolved:
                return index
        raise RuntimeError("Resolved feature was not found in the sequence feature list.")

    @staticmethod
    def _validate_selection_strand(strand: str | None) -> str | None:
        """Validate an optional selection strand."""

        if strand is None:
            return None
        if strand not in {"+", "-"}:
            raise ValueError("strand must be '+', '-', or None.")
        return strand

    @staticmethod
    def _feature_selection_strand(feature: Feature) -> str:
        """Return the strand to use when selecting a feature by annotation."""

        return feature.strand if feature.strand in {"+", "-"} else "+"

    def resolve_center(self, center: int | str | Feature) -> int:
        """Resolve an integer, feature name, or feature object to a sequence coordinate."""

        if isinstance(center, int):
            return center
        if isinstance(center, Feature):
            return (center.start + center.end) // 2
        if center.startswith("feature:"):
            center = center.split(":", 1)[1]
        feature = self.feature(center, required=True)
        assert feature is not None
        return (feature.start + feature.end) // 2

    def window(
        self,
        center: int | str | Feature,
        size: int,
        *,
        name: str | None = None,
        pad: str = "N",
    ) -> "AnnotatedSequence":
        """Return a fixed-size window centered on ``center``."""

        if size <= 0:
            raise ValueError("size must be positive")
        center_pos = self.resolve_center(center)
        start = center_pos - size // 2
        end = start + size
        clip_start = max(0, start)
        clip_end = min(len(self), end)
        left_pad = pad * max(0, -start)
        right_pad = pad * max(0, end - len(self))
        clipped = self[clip_start:clip_end]
        assert isinstance(clipped, AnnotatedSequence)
        shifted_features = tuple(feature.shift(len(left_pad)) for feature in clipped.features)
        features = shifted_features
        if left_pad:
            features += (Feature("left_padding", 0, len(left_pad), type="padding", source="padding"),)
        if right_pad:
            features += (
                Feature(
                    "right_padding",
                    len(left_pad) + len(clipped.sequence),
                    size,
                    type="padding",
                    source="padding",
                ),
            )
        return AnnotatedSequence(
            left_pad + clipped.sequence + right_pad,
            name=name or self.name,
            features=features,
            coordinate_map=clipped.coordinate_map,
            metadata={**dict(self.metadata), "window_center": center_pos, "window_start": start},
        )

    def reverse_complement(self, name: str | None = None) -> "AnnotatedSequence":
        """Return the reverse complement with features remapped."""

        length = len(self)
        features = []
        for feature in reversed(self.features):
            strand = {"+": "-", "-": "+"}.get(feature.strand, feature.strand)
            features.append(
                Feature(
                    feature.name,
                    length - feature.end,
                    length - feature.start,
                    type=feature.type,
                    strand=strand,
                    source=feature.source,
                    metadata=feature.metadata,
                )
            )
        return AnnotatedSequence(
            self.sequence.translate(_COMPLEMENT)[::-1].upper(),
            name=name or self.name,
            features=tuple(features),
            coordinate_map=self.coordinate_map.reverse_complement(length),
            metadata={**dict(self.metadata), "reverse_complemented": True},
        )

    @staticmethod
    def concat(*parts: "AnnotatedSequence", name: str | None = None) -> "AnnotatedSequence":
        """Concatenate annotated sequences and shift their annotations."""

        sequence_parts: list[str] = []
        features: list[Feature] = []
        coordinate_map = CoordinateMap(())
        offset = 0
        for part in parts:
            sequence_parts.append(part.sequence)
            features.extend(feature.shift(offset) for feature in part.features)
            coordinate_map = coordinate_map.concat(part.coordinate_map, offset)
            offset += len(part)
        return AnnotatedSequence("".join(sequence_parts), name=name, features=features, coordinate_map=coordinate_map)

    def replace(
        self,
        start: int,
        end: int,
        replacement: str | "AnnotatedSequence",
        *,
        preserve_partial_features: bool = False,
    ) -> "AnnotatedSequence":
        """Return a sequence where ``[start, end)`` is replaced by new bases.

        By default, features that overlap the replaced span are removed because
        their biological meaning may no longer be valid. Set
        ``preserve_partial_features=True`` when the replacement is a known
        construct edit and surviving feature pieces should remain annotated.
        """

        replacement_sequence = replacement.sequence if isinstance(replacement, AnnotatedSequence) else str(replacement).upper()
        new_sequence = self.sequence[:start] + replacement_sequence + self.sequence[end:]
        delta = len(replacement_sequence) - (end - start)
        features: list[Feature] = []
        for feature in self.features:
            if feature.end <= start:
                features.append(feature)
            elif feature.start >= end:
                features.append(feature.shift(delta))
            elif preserve_partial_features:
                features.extend(
                    self._preserved_feature_pieces(
                        feature,
                        start=start,
                        end=end,
                        delta=delta,
                    )
                )
        if isinstance(replacement, AnnotatedSequence):
            features.extend(feature.shift(start) for feature in replacement.features)
        else:
            features.append(Feature("replacement", start, start + len(replacement_sequence), type="replacement"))
        return AnnotatedSequence(new_sequence, name=self.name, features=features, metadata=self.metadata)

    @staticmethod
    def _preserved_feature_pieces(
        feature: Feature,
        *,
        start: int,
        end: int,
        delta: int,
    ) -> list[Feature]:
        """Return surviving pieces of a feature clipped by replacement."""

        pieces: list[Feature] = []
        base_metadata = {
            **dict(feature.metadata or {}),
            "clipped_by_replacement": True,
            "original_start": feature.start,
            "original_end": feature.end,
            "replacement_start": start,
            "replacement_end": end,
        }

        if feature.start < start and end < feature.end:
            pieces.append(
                replace(
                    feature,
                    end=feature.end + delta,
                    metadata={**base_metadata, "replacement_clip_side": "spanning"},
                )
            )
            return pieces

        if feature.start < start:
            left_end = min(feature.end, start)
            if left_end > feature.start:
                pieces.append(
                    replace(
                        feature,
                        end=left_end,
                        metadata={**base_metadata, "replacement_clip_side": "left"},
                    )
                )

        if feature.end > end:
            right_start = max(feature.start, end)
            if feature.end > right_start:
                pieces.append(
                    replace(
                        feature,
                        start=right_start + delta,
                        end=feature.end + delta,
                        metadata={**base_metadata, "replacement_clip_side": "right"},
                    )
                )

        return pieces

    def with_metadata(self, **metadata: Any) -> "AnnotatedSequence":
        """Return a copy with merged metadata."""

        return replace(self, metadata={**dict(self.metadata), **metadata})

    def to_fasta(self, header: str | None = None) -> str:
        """Return this sequence in FASTA format."""

        header = header or self.name or "sequence"
        lines = [f">{header}"]
        lines.extend(self.sequence[i : i + 80] for i in range(0, len(self.sequence), 80))
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the sequence and annotations."""

        return {
            "name": self.name,
            "sequence": self.sequence,
            "features": [feature.to_dict() for feature in self.features],
            "metadata": dict(self.metadata),
        }

    def feature_frame(self):
        """Return sequence features as a pandas DataFrame."""

        import pandas as pd

        return pd.DataFrame([feature.to_dict() for feature in self.features])

    def plot_annotations(
        self,
        *,
        start: int | None = None,
        end: int | None = None,
        types: Sequence[str] | None = None,
        hide_types: Sequence[str] | None = ("source",),
        title: str | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] = (12.0, 2.8),
        label_features: bool = True,
        show_legend: bool = True,
        save_path: str | Path | None = None,
    ):
        """Plot feature intervals on this sequence.

        The plot is intentionally simple: it is a debugging view for checking
        whether context builders, plasmid insertion, windowing, and padding kept
        annotations where you expect them.

        GenBank ``source`` features often span most or all of a plasmid and
        overlap every smaller annotation. They are hidden by default because
        they tend to dominate plasmid-context debug plots; pass
        ``hide_types=()`` to show the raw annotation set.
        """

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        plot_start = 0 if start is None else int(start)
        plot_end = len(self) if end is None else int(end)
        if plot_end <= plot_start:
            raise ValueError("end must be greater than start")

        type_filter = {value.lower() for value in types} if types is not None else None
        hidden_types = {value.lower() for value in hide_types} if hide_types is not None else set()
        features = [
            feature
            for feature in self.features
            if feature.overlaps(plot_start, plot_end)
            and (type_filter is None or feature.type.lower() in type_filter)
            and feature.type.lower() not in hidden_types
        ]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        fig = ax.figure

        ax.hlines(0.0, plot_start, plot_end, color="black", linewidth=1.0, alpha=0.45)
        lanes: list[int] = []
        palette = plt.get_cmap("tab20")
        type_to_color: dict[str, Any] = {}

        for feature in sorted(features, key=lambda item: (item.start, item.end, item.type)):
            local_start = max(feature.start, plot_start)
            local_end = min(feature.end, plot_end)
            lane = 0
            while lane < len(lanes) and local_start < lanes[lane]:
                lane += 1
            if lane == len(lanes):
                lanes.append(local_end)
            else:
                lanes[lane] = local_end

            color = type_to_color.setdefault(feature.type, palette(len(type_to_color) % palette.N))
            y = lane * 0.55 + 0.18
            ax.add_patch(
                Rectangle(
                    (local_start, y),
                    max(1, local_end - local_start),
                    0.34,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.78,
                )
            )
            if label_features:
                label = feature.name if feature.name != feature.type else feature.type
                ax.text(
                    (local_start + local_end) / 2,
                    y + 0.18,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    clip_on=True,
                )

        ax.set_xlim(plot_start, plot_end)
        ax.set_ylim(-0.25, max(0.9, len(lanes) * 0.55 + 0.75))
        ax.set_yticks([])
        ax.set_xlabel("Sequence coordinate")
        ax.set_title(title or self.name or "Annotated sequence")
        if show_legend and type_to_color:
            handles = [
                Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5, alpha=0.78)
                for _, color in type_to_color.items()
            ]
            ax.legend(
                handles,
                list(type_to_color),
                loc="upper right",
                frameon=False,
                fontsize=8,
                ncol=min(4, len(type_to_color)),
            )
        ax.spines[["left", "right", "top"]].set_visible(False)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig, ax

@dataclass(frozen=True)
class _PairCoordinateMapper:
    """Cached coordinate transform for one reference/alternative replacement."""

    ref_start: int
    ref_end: int
    alt_start: int
    alt_end: int

    @staticmethod
    def _map_boundary(
        position: int,
        *,
        source_start: int,
        source_end: int,
        target_start: int,
        target_end: int,
        side: Literal["left", "right"],
    ) -> int:
        if position <= source_start:
            return target_start + (position - source_start)
        if position >= source_end:
            return target_end + (position - source_end)

        source_length = source_end - source_start
        target_length = target_end - target_start
        if source_length <= 0:
            return target_start if side == "left" else target_end
        numerator = (position - source_start) * target_length
        if side == "left":
            offset = numerator // source_length
        else:
            offset = (numerator + source_length - 1) // source_length
        return target_start + offset

    def map_ref_interval_to_alt(self, start: int, end: int) -> tuple[int, int]:
        """Map a reference-local half-open interval to alternative coordinates."""

        if end < start:
            raise ValueError("end must be greater than or equal to start")
        mapped_start = self._map_boundary(
            int(start),
            source_start=self.ref_start,
            source_end=self.ref_end,
            target_start=self.alt_start,
            target_end=self.alt_end,
            side="left",
        )
        mapped_end = self._map_boundary(
            int(end),
            source_start=self.ref_start,
            source_end=self.ref_end,
            target_start=self.alt_start,
            target_end=self.alt_end,
            side="right",
        )
        return mapped_start, max(mapped_start, mapped_end)

    def map_alt_interval_to_ref(self, start: int, end: int) -> tuple[int, int]:
        """Map an alternative-local half-open interval to reference coordinates."""

        if end < start:
            raise ValueError("end must be greater than or equal to start")
        mapped_start = self._map_boundary(
            int(start),
            source_start=self.alt_start,
            source_end=self.alt_end,
            target_start=self.ref_start,
            target_end=self.ref_end,
            side="left",
        )
        mapped_end = self._map_boundary(
            int(end),
            source_start=self.alt_start,
            source_end=self.alt_end,
            target_start=self.ref_start,
            target_end=self.ref_end,
            side="right",
        )
        return mapped_start, max(mapped_start, mapped_end)


@dataclass(frozen=True)
class SequencePair:
    """Reference and alternative annotated sequences for one variant."""

    ref: AnnotatedSequence
    alt: AnnotatedSequence
    variant: Any | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def variant_feature(self) -> Feature | None:
        """Return the variant feature from the reference or alternative sequence."""

        return self.ref.feature("variant", required=False) or self.alt.feature("variant", required=False)

    def variant_features(self) -> tuple[Feature | None, Feature | None]:
        """Return variant features from reference and alternative sequences."""

        return self.ref.feature("variant", required=False), self.alt.feature("variant", required=False)

    def assert_compatible(self) -> None:
        """Raise when the reference and alternative sequences are clearly incompatible."""

        if not self.ref.sequence or not self.alt.sequence:
            raise ValueError("Reference and alternative sequences must be non-empty.")

    def map(self, fn: Callable[[AnnotatedSequence], AnnotatedSequence]) -> "SequencePair":
        """Apply ``fn`` to both sequences and return a new sequence pair.

        ``fn`` must accept one :class:`AnnotatedSequence` and return an
        :class:`AnnotatedSequence`. The returned pair keeps the same variant and
        metadata, which makes this useful for applying the same context,
        trimming, masking, or annotation transform to reference and alternative
        sequences.
        """

        ref = fn(self.ref)
        alt = fn(self.alt)
        if not isinstance(ref, AnnotatedSequence) or not isinstance(alt, AnnotatedSequence):
            raise TypeError("SequencePair.map() callable must return AnnotatedSequence objects.")
        pair = SequencePair(ref=ref, alt=alt, variant=self.variant, metadata=self.metadata)
        pair.assert_compatible()
        return pair

    def to_dict(self) -> dict[str, Any]:
        """Serialize the sequence pair."""

        return {
            "ref": self.ref.to_dict(),
            "alt": self.alt.to_dict(),
            "variant": self.variant.to_dict() if hasattr(self.variant, "to_dict") else self.variant,
            "metadata": dict(self.metadata),
        }

    def _variant_alleles(self) -> tuple[str, str] | None:
        """Return the explicit reference/alternative alleles, when available."""

        if self.variant is None:
            return None
        if isinstance(self.variant, Mapping):
            if "ref" in self.variant and "alt" in self.variant:
                return str(self.variant["ref"]), str(self.variant["alt"])
            return None
        ref = getattr(self.variant, "ref", None)
        alt = getattr(self.variant, "alt", None)
        if ref is None or alt is None:
            return None
        return str(ref), str(alt)

    def _exact_difference_interval(self, *, method: str = "auto") -> tuple[int, int, int, int]:
        """Return changed intervals using exact allele lengths when possible.

        Variant annotations use a one-base visual anchor for empty alleles. That
        is useful for annotation plots, but it is not the true length of an
        insertion or deletion. Coordinate mapping and gapped sequence rendering
        therefore use the explicit variant allele lengths when they are known.
        """

        ref_start, ref_end, alt_start, alt_end = self.difference_interval(method=method)
        alleles = self._variant_alleles()
        if method == "string" or alleles is None:
            return ref_start, ref_end, alt_start, alt_end
        ref_allele, alt_allele = alleles
        ref_feature, alt_feature = self.variant_features()
        if ref_feature is None and alt_feature is None:
            # Prefix/suffix comparison already produced exact local intervals;
            # without an annotation anchor there is no reliable way to place
            # externally supplied alleles back into the sequence.
            return ref_start, ref_end, alt_start, alt_end
        if ref_feature is not None:
            ref_start = ref_feature.start
        if alt_feature is not None:
            alt_start = alt_feature.start

        # Use the alleles as they actually occur in each local sequence. This
        # keeps prefix/suffix normalization correct for reverse-complemented
        # contexts while explicit allele lengths still recover empty alleles.
        ref_allele = self.ref.sequence[ref_start : ref_start + len(ref_allele)]
        alt_allele = self.alt.sequence[alt_start : alt_start + len(alt_allele)]

        prefix_length = 0
        shared_limit = min(len(ref_allele), len(alt_allele))
        while prefix_length < shared_limit and ref_allele[prefix_length] == alt_allele[prefix_length]:
            prefix_length += 1
        suffix_length = 0
        suffix_limit = shared_limit - prefix_length
        while (
            suffix_length < suffix_limit
            and ref_allele[len(ref_allele) - suffix_length - 1]
            == alt_allele[len(alt_allele) - suffix_length - 1]
        ):
            suffix_length += 1

        ref_core_end = len(ref_allele) - suffix_length if suffix_length else len(ref_allele)
        alt_core_end = len(alt_allele) - suffix_length if suffix_length else len(alt_allele)
        return (
            ref_start + prefix_length,
            ref_start + ref_core_end,
            alt_start + prefix_length,
            alt_start + alt_core_end,
        )

    def coordinate_mapper(self) -> _PairCoordinateMapper:
        """Return one cached reference/alternative coordinate transform."""

        return _PairCoordinateMapper(*self._exact_difference_interval())

    def map_ref_interval_to_alt(self, start: int, end: int) -> tuple[int, int]:
        """Map a reference-local half-open interval to alternative coordinates."""

        return self.coordinate_mapper().map_ref_interval_to_alt(start, end)

    def map_alt_interval_to_ref(self, start: int, end: int) -> tuple[int, int]:
        """Map an alternative-local half-open interval to reference coordinates."""

        return self.coordinate_mapper().map_alt_interval_to_ref(start, end)

    def difference_interval(self, *, method: str = "auto") -> tuple[int, int, int, int]:
        """Return changed intervals as ``ref_start, ref_end, alt_start, alt_end``.

        ``method="auto"`` uses explicit ``variant`` annotations when present,
        which is robust to circular-context rotations. ``method="string"``
        preserves the old first/last mismatch behavior.
        """

        if method not in {"auto", "feature", "string"}:
            raise ValueError("method must be 'auto', 'feature', or 'string'.")

        ref_variant, alt_variant = self.variant_features()
        if method in {"auto", "feature"} and (ref_variant is not None or alt_variant is not None):
            if ref_variant is None:
                assert alt_variant is not None
                ref_variant = alt_variant
            if alt_variant is None:
                alt_variant = ref_variant
            return ref_variant.start, ref_variant.end, alt_variant.start, alt_variant.end
        if method == "feature":
            raise ValueError("No variant feature is available on either sequence.")

        ref = self.ref.sequence
        alt = self.alt.sequence
        min_len = min(len(ref), len(alt))
        left = 0
        while left < min_len and ref[left] == alt[left]:
            left += 1

        right_ref = len(ref)
        right_alt = len(alt)
        while right_ref > left and right_alt > left and ref[right_ref - 1] == alt[right_alt - 1]:
            right_ref -= 1
            right_alt -= 1
        return left, right_ref, left, right_alt

    def plot_difference(
        self,
        *,
        flank: int = 40,
        method: str = "auto",
        show_gaps: bool = True,
        gap_char: str = "-",
        ax: Any | None = None,
        figsize: tuple[float, float] = (12.0, 2.4),
        title: str | None = None,
        save_path: str | Path | None = None,
    ):
        """Plot the exact changed sequence segment between reference and alternative.

        Unequal alleles are aligned around their shared prefix/suffix and the
        shorter changed segment is padded with ``gap_char``. Gap insertion is a
        display operation only and never changes either input sequence.
        """

        import matplotlib.pyplot as plt

        if len(gap_char) != 1 or gap_char.isspace():
            raise ValueError("gap_char must be one visible character")

        ref_start, ref_end, alt_start, alt_end = self._exact_difference_interval(method=method)
        left = max(0, ref_start - flank)
        right = min(len(self.ref), ref_end + flank)
        alt_left = max(0, alt_start - flank)
        alt_right = min(len(self.alt), alt_end + flank)

        ref_left = self.ref.sequence[left:ref_start]
        alt_left_text = self.alt.sequence[alt_left:alt_start]
        ref_allele = self.ref.sequence[ref_start:ref_end]
        alt_allele = self.alt.sequence[alt_start:alt_end]
        ref_right = self.ref.sequence[ref_end:right]
        alt_right_text = self.alt.sequence[alt_end:alt_right]

        prefix_length = 0
        shared_limit = min(len(ref_allele), len(alt_allele))
        while prefix_length < shared_limit and ref_allele[prefix_length] == alt_allele[prefix_length]:
            prefix_length += 1

        suffix_length = 0
        suffix_limit = shared_limit - prefix_length
        while (
            suffix_length < suffix_limit
            and ref_allele[len(ref_allele) - suffix_length - 1]
            == alt_allele[len(alt_allele) - suffix_length - 1]
        ):
            suffix_length += 1

        ref_core_end = len(ref_allele) - suffix_length if suffix_length else len(ref_allele)
        alt_core_end = len(alt_allele) - suffix_length if suffix_length else len(alt_allele)
        ref_core = ref_allele[prefix_length:ref_core_end]
        alt_core = alt_allele[prefix_length:alt_core_end]
        core_width = max(len(ref_core), len(alt_core), 1)

        if show_gaps:
            ref_aligned_allele = (
                ref_allele[:prefix_length]
                + ref_core.ljust(core_width, gap_char)
                + (ref_allele[ref_core_end:] if suffix_length else "")
            )
            alt_aligned_allele = (
                alt_allele[:prefix_length]
                + alt_core.ljust(core_width, gap_char)
                + (alt_allele[alt_core_end:] if suffix_length else "")
            )
        else:
            ref_aligned_allele = ref_allele
            alt_aligned_allele = alt_allele
            core_width = max(len(ref_core), len(alt_core), 1)

        left_width = max(len(ref_left), len(alt_left_text))
        right_width = max(len(ref_right), len(alt_right_text))
        ref_text = (
            ref_left.rjust(left_width)
            + ref_aligned_allele
            + ref_right.ljust(right_width)
        )
        alt_text = (
            alt_left_text.rjust(left_width)
            + alt_aligned_allele
            + alt_right_text.ljust(right_width)
        )
        marker_start = left_width + prefix_length
        marker_width = core_width

        if len(ref_text) > 140:
            ref_text = ref_text[:137] + "..."
        if len(alt_text) > 140:
            alt_text = alt_text[:137] + "..."
        visible_marker_width = max(1, min(marker_width, max(1, len(ref_text) - marker_start)))

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        fig = ax.figure
        ax.axis("off")

        ax.text(0.01, 0.72, "REF", weight="bold", ha="left", va="center", transform=ax.transAxes)
        ax.text(0.01, 0.38, "ALT", weight="bold", ha="left", va="center", transform=ax.transAxes)
        ax.text(0.09, 0.72, ref_text, family="monospace", ha="left", va="center", transform=ax.transAxes)
        ax.text(0.09, 0.38, alt_text, family="monospace", ha="left", va="center", transform=ax.transAxes)
        ax.text(
            0.09,
            0.57,
            " " * marker_start + "^" * visible_marker_width,
            family="monospace",
            color="crimson",
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.09,
            0.23,
            " " * marker_start + "^" * visible_marker_width,
            family="monospace",
            color="crimson",
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

        summary = (
            f"ref[{ref_start}:{ref_end}]={self.ref.sequence[ref_start:ref_end]!r} -> "
            f"alt[{alt_start}:{alt_end}]={self.alt.sequence[alt_start:alt_end]!r}"
        )
        ax.text(0.09, 0.08, summary, fontsize=9, ha="left", va="center", transform=ax.transAxes)
        ax.set_title(title or "Variant sequence difference")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight")
        return fig, ax
