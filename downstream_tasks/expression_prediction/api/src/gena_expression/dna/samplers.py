"""Sequence samplers for genomic regions."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from ..sequences import AnnotatedSequence
from .genome import Genome


DEFAULT_TRANSCRIPT_FEATURE_TYPES = (
    "transcript",
    "mrna",
    "ncrna",
    "lnc_rna",
    "lncrna",
    "rrna",
    "trna",
    "snrna",
    "snorna",
    "mirna",
    "primary_transcript",
    "pseudogenic_transcript",
)


@dataclass(frozen=True, slots=True)
class _TssWindow:
    """One clipped genomic sampling window around a unique TSS."""

    chrom: str
    start: int
    end: int
    tss: int
    strand: str
    transcript_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    gene_names: tuple[str, ...]
    feature_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _WindowGroup:
    """One connected component of overlapping TSS windows."""

    chrom: str
    start: int
    end: int
    windows: tuple[_TssWindow, ...]


@dataclass(slots=True)
class _TssAccumulator:
    """Mutable metadata collected for transcript records sharing one TSS."""

    start: int
    end: int
    transcript_ids: set[str] = field(default_factory=set)
    gene_ids: set[str] = field(default_factory=set)
    gene_names: set[str] = field(default_factory=set)
    feature_types: set[str] = field(default_factory=set)


class PromoterSampler(Iterator[AnnotatedSequence]):
    """Yield random transcript-oriented sequences from non-overlapping TSS regions.

    Transcript TSS windows that overlap on the same chromosome are grouped into
    one sampling unit. Each unit is visited once per cycle. On every visit, one
    unique TSS window in the unit and one valid sequence start within that
    window are selected randomly.

    ``tss_sampling_window`` is the total width of the genomic window centered
    on each TSS. Returned sequences use the transcript strand, so sequences for
    negative-strand transcripts are reverse-complemented.
    """

    def __init__(
        self,
        genome: Genome,
        seq_len: int,
        tss_sampling_window: int,
        *,
        seed: int | None = None,
        transcript_feature_types: Sequence[str] | str | None = None,
        include_features: bool = True,
    ) -> None:
        """Find transcript TSS windows and configure reproducible sampling."""

        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if (
            not isinstance(tss_sampling_window, int)
            or isinstance(tss_sampling_window, bool)
            or tss_sampling_window <= 0
        ):
            raise ValueError("tss_sampling_window must be a positive integer")
        if tss_sampling_window < seq_len:
            raise ValueError("tss_sampling_window must be greater than or equal to seq_len")

        if transcript_feature_types is None:
            transcript_feature_types = DEFAULT_TRANSCRIPT_FEATURE_TYPES
        elif isinstance(transcript_feature_types, str):
            transcript_feature_types = (transcript_feature_types,)
        normalized_types = frozenset(value.lower() for value in transcript_feature_types)
        if not normalized_types:
            raise ValueError("transcript_feature_types must not be empty")

        self.genome = genome
        self.seq_len = seq_len
        self.tss_sampling_window = tss_sampling_window
        self.transcript_feature_types = tuple(sorted(normalized_types))
        self.include_features = bool(include_features)
        self._rng = random.Random(seed)
        self._groups = self._build_groups(normalized_types)
        self._next_group = 0
        self._cycle = 0

    def __iter__(self) -> PromoterSampler:
        """Return this infinite sampler iterator."""

        return self

    def __len__(self) -> int:
        """Return the number of non-overlapping sampling units."""

        return len(self._groups)

    def __next__(self) -> AnnotatedSequence:
        """Sample one sequence from the next non-overlapping TSS region."""

        group_index = self._next_group
        cycle = self._cycle
        group = self._groups[group_index]
        window = self._rng.choice(group.windows)
        start = self._rng.randint(window.start, window.end - self.seq_len)
        end = start + self.seq_len

        label = (
            window.transcript_ids[0]
            if window.transcript_ids
            else window.gene_names[0]
            if window.gene_names
            else f"tss_{window.tss}"
        )
        sequence = self.genome.sequence(
            window.chrom,
            start,
            end,
            strand=window.strand,
            include_features=self.include_features,
            name=f"{label}|{window.chrom}:{start}-{end}({window.strand})",
        )

        if window.strand == "+":
            local_tss = window.tss - start
        else:
            local_tss = end - 1 - window.tss
        tss_in_sequence = 0 <= local_tss < len(sequence)
        if tss_in_sequence:
            sequence = sequence.add_feature(
                "tss",
                local_tss,
                local_tss + 1,
                type="tss",
                source="annotation",
                metadata={"chrom": window.chrom, "position": window.tss, "strand": window.strand},
            )

        sequence = sequence.with_metadata(
            promoter_group_index=group_index,
            promoter_cycle=cycle,
            promoter_group_start=group.start,
            promoter_group_end=group.end,
            sampling_window_start=window.start,
            sampling_window_end=window.end,
            genomic_start=start,
            genomic_end=end,
            chrom=window.chrom,
            strand=window.strand,
            tss=window.tss,
            tss_in_sequence=tss_in_sequence,
            transcript_ids=window.transcript_ids,
            gene_ids=window.gene_ids,
            gene_names=window.gene_names,
            transcript_feature_types=window.feature_types,
        )

        self._next_group += 1
        if self._next_group == len(self._groups):
            self._next_group = 0
            self._cycle += 1
        return sequence

    def _build_groups(self, feature_types: frozenset[str]) -> tuple[_WindowGroup, ...]:
        """Discover, deduplicate, clip, and group transcript TSS windows."""

        annotation = getattr(self.genome, "_annotation", None)
        if annotation is None:
            raise ValueError("PromoterSampler requires a Genome with an annotation")
        iter_records = getattr(annotation, "iter_records", None)
        if not callable(iter_records):
            raise TypeError("The configured genome annotation does not support record iteration")

        records_seen = 0
        transcript_records = 0
        invalid_strand = 0
        invalid_interval = 0
        missing_contig = 0
        too_short = 0
        chrom_names: dict[str, str] = {}
        chrom_lengths: dict[str, int] = {}
        windows_by_tss: dict[tuple[str, int, str], _TssAccumulator] = {}

        for record in iter_records():
            records_seen += 1
            if record.feature_type.lower() not in feature_types:
                continue
            transcript_records += 1
            if record.strand not in {"+", "-"}:
                invalid_strand += 1
                continue
            if record.end <= record.start:
                invalid_interval += 1
                continue

            if record.chrom not in chrom_names:
                try:
                    chrom_names[record.chrom] = self.genome.normalize_chrom(record.chrom)
                except ValueError:
                    chrom_names[record.chrom] = ""
            chrom = chrom_names[record.chrom]
            if not chrom:
                missing_contig += 1
                continue
            if chrom not in chrom_lengths:
                chrom_lengths[chrom] = self.genome.chrom_length(chrom)

            tss = record.start if record.strand == "+" else record.end - 1
            raw_start = tss - self.tss_sampling_window // 2
            raw_end = raw_start + self.tss_sampling_window
            start = max(0, raw_start)
            end = min(chrom_lengths[chrom], raw_end)
            if end - start < self.seq_len:
                too_short += 1
                continue

            key = (chrom, tss, record.strand)
            values = windows_by_tss.setdefault(
                key,
                _TssAccumulator(start=start, end=end),
            )
            attributes = record.attributes
            transcript_id = (
                attributes.get("transcript_id")
                or attributes.get("ID")
                or attributes.get("transcript_name")
                or attributes.get("Name")
            )
            gene_id = attributes.get("gene_id") or attributes.get("gene") or attributes.get("Parent")
            gene_name = attributes.get("gene_name")
            if transcript_id:
                values.transcript_ids.add(transcript_id)
            if gene_id:
                values.gene_ids.add(gene_id)
            if gene_name:
                values.gene_names.add(gene_name)
            values.feature_types.add(record.feature_type)

        windows = [
            _TssWindow(
                chrom=chrom,
                start=values.start,
                end=values.end,
                tss=tss,
                strand=strand,
                transcript_ids=tuple(sorted(values.transcript_ids)),
                gene_ids=tuple(sorted(values.gene_ids)),
                gene_names=tuple(sorted(values.gene_names)),
                feature_types=tuple(sorted(values.feature_types)),
            )
            for (chrom, tss, strand), values in windows_by_tss.items()
        ]
        windows.sort(key=lambda window: (window.chrom, window.start, window.end, window.tss, window.strand))

        if not windows:
            raise ValueError(
                "PromoterSampler found no usable transcript TSS windows. "
                f"debug={{'records_seen': {records_seen}, 'transcript_records': {transcript_records}, "
                f"'invalid_strand': {invalid_strand}, 'invalid_interval': {invalid_interval}, "
                f"'missing_contig': {missing_contig}, 'too_short': {too_short}}}"
            )

        groups: list[_WindowGroup] = []
        members: list[_TssWindow] = []
        group_chrom = ""
        group_start = 0
        group_end = 0
        for window in windows:
            if members and window.chrom == group_chrom and window.start < group_end:
                members.append(window)
                group_end = max(group_end, window.end)
                continue
            if members:
                groups.append(_WindowGroup(group_chrom, group_start, group_end, tuple(members)))
            members = [window]
            group_chrom = window.chrom
            group_start = window.start
            group_end = window.end
        groups.append(_WindowGroup(group_chrom, group_start, group_end, tuple(members)))
        return tuple(groups)
