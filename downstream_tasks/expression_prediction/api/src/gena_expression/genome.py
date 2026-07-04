"""Small FASTA-backed genome adapter."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from .sequences import AnnotatedSequence, CoordinateMap, CoordinateSegment, Feature


@dataclass(frozen=True)
class _GtfRecord:
    """One parsed GTF row in internal 0-based half-open coordinates."""

    chrom: str
    source: str
    feature_type: str
    start: int
    end: int
    score: str
    strand: str
    frame: str
    attributes: Mapping[str, str]


class Genome:
    """FASTA-backed genome object with chromosome-name normalization.

    ``annotation`` may be a GTF path. GTF coordinates are converted from
    1-based inclusive to internal 0-based half-open coordinates.
    """

    def __init__(
        self,
        fasta_path: str | Path,
        annotation: str | Path | object | None = None,
        build: str | None = None,
        chrom_style: Literal["auto", "chr", "no_chr"] = "auto",
        cache: bool = True,
    ) -> None:
        self.fasta_path = Path(fasta_path)
        self.annotation = annotation
        self.build = build
        self.chrom_style = chrom_style
        self.cache = cache
        self._fasta = None
        self._annotation_by_chrom: dict[str, list[_GtfRecord]] = {}
        if annotation is not None:
            self._annotation_by_chrom = self._load_annotation(annotation)

    def _handle(self):
        import pysam

        if not self.cache:
            return pysam.FastaFile(str(self.fasta_path))
        if self._fasta is None:
            self._fasta = pysam.FastaFile(str(self.fasta_path))
        return self._fasta

    def fetch(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        strand: str = "+",
        pad: str = "N",
    ) -> str:
        """Fetch a 0-based half-open sequence interval, padding out-of-bounds."""

        fasta = self._handle()
        reference = self.normalize_chrom(chrom)
        chrom_len = fasta.get_reference_length(reference)
        left_pad = pad * max(0, -start)
        right_pad = pad * max(0, end - chrom_len)
        fetch_start = max(0, start)
        fetch_end = min(chrom_len, end)
        sequence = fasta.fetch(reference, fetch_start, fetch_end).upper()
        sequence = left_pad + sequence + right_pad
        if strand == "-":
            table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
            sequence = sequence.translate(table)[::-1].upper()
        return sequence

    def sequence(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        strand: str = "+",
        include_features: bool = True,
        name: str | None = None,
    ) -> AnnotatedSequence:
        """Fetch an interval as an :class:`AnnotatedSequence`."""

        normalized = self.normalize_chrom(chrom)
        sequence = self.fetch(normalized, start, end, strand="+")
        cmap = CoordinateMap(
            (
                CoordinateSegment(
                    0,
                    len(sequence),
                    source="genome",
                    chrom=normalized,
                    source_start=max(0, start),
                    source_end=max(0, start) + len(sequence),
                    strand="+",
                    metadata={"build": self.build},
                ),
            )
        )
        features = []
        if include_features:
            features.append(
                Feature(
                    name=f"{normalized}:{start}-{end}",
                    start=0,
                    end=len(sequence),
                    type="genome_interval",
                    strand="+",
                    source="genome",
                    metadata={"chrom": normalized, "start": start, "end": end, "build": self.build},
                )
            )
            features.extend(self.features(normalized, start, end))
        seq = AnnotatedSequence(sequence, name=name, features=features, coordinate_map=cmap)
        if strand == "-":
            seq = seq.reverse_complement(name=name)
        return seq

    def sequence_around(
        self,
        chrom: str,
        center: int,
        size: int,
        *,
        strand: str = "+",
        include_features: bool = True,
        center_feature_name: str | None = None,
        name: str | None = None,
    ) -> AnnotatedSequence:
        """Fetch a fixed-size sequence centered on a genomic coordinate."""

        start = int(center) - int(size) // 2
        seq = self.sequence(chrom, start, start + int(size), strand=strand, include_features=include_features, name=name)
        if center_feature_name is not None:
            seq = seq.add_feature(center_feature_name, int(size) // 2, int(size) // 2 + 1, type=center_feature_name)
        return seq

    def features(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        types: Sequence[str] | None = None,
        strand: str | None = None,
    ) -> list[Feature]:
        """Return annotation features overlapping ``[start, end)``.

        Returned feature coordinates are relative to the requested interval, so
        they can be attached directly to the corresponding
        :class:`AnnotatedSequence`.
        """

        if not self._annotation_by_chrom:
            return []

        normalized = self._normalize_annotation_chrom(chrom)
        records = self._annotation_by_chrom.get(normalized, [])
        type_filter = {value.lower() for value in types} if types is not None else None
        features: list[Feature] = []

        for record in records:
            if record.end <= start:
                continue
            if record.start >= end:
                break
            if type_filter is not None and record.feature_type.lower() not in type_filter:
                continue
            if strand is not None and strand != record.strand:
                continue

            clipped_start = max(record.start, start)
            clipped_end = min(record.end, end)
            if clipped_end <= clipped_start:
                continue

            feature_name = (
                record.attributes.get("gene_name")
                or record.attributes.get("transcript_name")
                or record.attributes.get("gene_id")
                or f"{record.feature_type}:{record.chrom}:{record.start}-{record.end}"
            )
            features.append(
                Feature(
                    name=feature_name,
                    start=clipped_start - start,
                    end=clipped_end - start,
                    type=record.feature_type,
                    strand=record.strand,
                    source=record.source,
                    metadata={
                        "chrom": record.chrom,
                        "genomic_start": record.start,
                        "genomic_end": record.end,
                        "score": record.score,
                        "frame": record.frame,
                        "attributes": dict(record.attributes),
                    },
                )
            )

        return features

    def validate_ref(
        self,
        chrom: str,
        pos: int,
        ref: str,
        *,
        coordinate_system: Literal["0-based", "1-based"] = "0-based",
    ) -> None:
        """Warn if ``ref`` does not match the FASTA at ``chrom:pos``."""

        pos0 = pos - 1 if coordinate_system == "1-based" else pos
        observed = self.fetch(chrom, pos0, pos0 + len(ref))
        if observed.upper() != ref.upper():
            warnings.warn(
                "Reference mismatch during genome validation. "
                f"debug={{'chrom': {chrom!r}, 'pos': {pos!r}, "
                f"'pos0': {pos0!r}, 'coordinate_system': {coordinate_system!r}, "
                f"'expected': {ref!r}, 'observed': {observed!r}}}",
                RuntimeWarning,
                stacklevel=2,
            )

    def normalize_chrom(self, chrom: str) -> str:
        """Map a chromosome label to one present in the FASTA index."""

        fasta = self._handle()
        names = set(fasta.references)
        raw = chrom.removeprefix("chr")
        candidates = [chrom]
        if self.chrom_style in {"auto", "chr"}:
            candidates.append(f"chr{raw}")
        if self.chrom_style in {"auto", "no_chr"}:
            candidates.append(raw)
        if raw in {"M", "MT"}:
            candidates.extend(["M", "MT", "chrM", "chrMT"])
        for candidate in dict.fromkeys(candidates):
            if candidate in names:
                return candidate
        raise ValueError(f"Chromosome {chrom!r} was not found in FASTA. Tried: {', '.join(candidates)}")

    def _normalize_annotation_chrom(self, chrom: str) -> str:
        """Map a chromosome label to one present in the GTF annotation index."""

        if chrom in self._annotation_by_chrom:
            return chrom
        raw = chrom.removeprefix("chr")
        candidates = [chrom, raw, f"chr{raw}"]
        if raw in {"M", "MT"}:
            candidates.extend(["M", "MT", "chrM", "chrMT"])
        for candidate in dict.fromkeys(candidates):
            if candidate in self._annotation_by_chrom:
                return candidate
        return chrom

    @classmethod
    def _load_annotation(cls, annotation: str | Path | object) -> dict[str, list[_GtfRecord]]:
        """Load a supported annotation object into a chromosome-indexed table."""

        if isinstance(annotation, (str, Path)):
            path = Path(annotation)
            if path.suffix.lower() not in {".gtf", ".gff", ".gff3"}:
                raise ValueError(f"Only GTF-like annotation files are supported, got: {path}")
            return cls._load_gtf(path)
        if isinstance(annotation, Mapping):
            return {
                str(chrom): sorted(list(records), key=lambda record: (record.start, record.end))
                for chrom, records in annotation.items()
            }
        raise TypeError("annotation must be a GTF path or a chromosome-to-record mapping.")

    @classmethod
    def _load_gtf(cls, path: Path) -> dict[str, list[_GtfRecord]]:
        """Parse a GTF file into records grouped by chromosome."""

        by_chrom: dict[str, list[_GtfRecord]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) != 9:
                    raise ValueError(f"Invalid GTF line {line_number} in {path}: expected 9 columns")

                chrom, source, feature_type, start_text, end_text, score, strand, frame, attrs_text = fields
                start0 = int(start_text) - 1
                end0 = int(end_text)
                if start0 < 0 or end0 < start0:
                    raise ValueError(f"Invalid GTF coordinates on line {line_number}: {start_text}-{end_text}")

                record = _GtfRecord(
                    chrom=chrom,
                    source=source,
                    feature_type=feature_type,
                    start=start0,
                    end=end0,
                    score=score,
                    strand=strand,
                    frame=frame,
                    attributes=cls._parse_gtf_attributes(attrs_text),
                )
                by_chrom.setdefault(chrom, []).append(record)

        for records in by_chrom.values():
            records.sort(key=lambda record: (record.start, record.end))
        return by_chrom

    @staticmethod
    def _parse_gtf_attributes(text: str) -> dict[str, str]:
        """Parse the ninth GTF column into a simple string dictionary."""

        attributes: dict[str, str] = {}
        for match in re.finditer(r'(\S+)\s+"([^"]*)"', text):
            attributes[match.group(1)] = match.group(2)

        # Numeric and unquoted fields occur in some GTF-like files.
        parsed_spans = [match.span() for match in re.finditer(r'(\S+)\s+"([^"]*)"', text)]
        leftovers = text
        for start, end in reversed(parsed_spans):
            leftovers = leftovers[:start] + leftovers[end:]
        for chunk in leftovers.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(None, 1)
            if len(parts) == 2:
                attributes.setdefault(parts[0], parts[1].strip().strip('"'))
        return attributes

    def chrom_length(self, chrom: str) -> int:
        """Return chromosome length from the FASTA index."""

        return self._handle().get_reference_length(self.normalize_chrom(chrom))

    def close(self) -> None:
        """Close the cached FASTA handle, if one is open."""

        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None


@dataclass(frozen=True)
class SafeHarborSite:
    """Named human safe-harbor interval in 1-based inclusive coordinates."""

    abbrev_name: str
    chromosome: str
    start: int
    end: int
    category: str = ""
    genes: str = ""


@dataclass(frozen=True)
class GenomeInterval:
    """Simple 1-based inclusive genome interval."""

    chromosome: str
    start: int
    end: int

    @property
    def insertion_index(self) -> int:
        """Return interval midpoint as a 0-based insertion index."""

        return ((self.start - 1) + self.end) // 2


class GenomeRegion:
    """Build annotated fixed-length genome contexts around safe-harbor sites."""

    SAFE_HARBORS: ClassVar[dict[str, SafeHarborSite]] = {
        "Dep.34": SafeHarborSite("Dep.34", "3", 96719524, 96721025, "intergenic", "RPL18AP8/RCC2P5"),
        "Dep.36": SafeHarborSite("Dep.36", "6", 130764851, 130765531, "intergenic", "TMEM200A/SMLR1"),
        "Ap.102": SafeHarborSite("Ap.102", "8", 128435359, 128437048, "intergenic", "LINC00824/CCDC26"),
        "Dep.3": SafeHarborSite("Dep.3", "7", 23218501, 23222555, "intergenic", "NUP42/GPNMB"),
        "Dep.33": SafeHarborSite("Dep.33", "5", 50957501, 50959300, "intergenic", "PARP8/ISL1"),
        "Dep.13": SafeHarborSite("Dep.13", "21", 32097568, 32098575, "intergenic", "HUNK/Mis18A"),
        "Dep.28": SafeHarborSite("Dep.28", "12", 40658392, 40660499, "intergenic", "LRRK2/CNTN1"),
        "Dep.22": SafeHarborSite("Dep.22", "X", 98175245, 98175812, "intergenic", "DIAPH2/PCDH19"),
        "AAVS1": SafeHarborSite("AAVS1", "19", 55112144, 55117873, "intronic", "PPP1R12C"),
        "Prot.181": SafeHarborSite("Prot.181", "10", 4652078, 4656006, "intronic", "MANCR"),
        "Dep.2": SafeHarborSite("Dep.2", "6", 39422982, 39425107, "intronic", "KIF6"),
        "Prot.176": SafeHarborSite("Prot.176", "7", 125056326, 125057364, "intronic", "POT1-AS1"),
        "Dep.1": SafeHarborSite("Dep.1", "9", 36852789, 36855146, "intronic", "PAX5"),
        "Prot.2": SafeHarborSite("Prot.2", "12", 41262344, 41264604, "intronic", "PDZRN4"),
        "Dep.35": SafeHarborSite("Dep.35", "7", 118262500, 118263554, "intronic", "ANKRD7"),
        "Dep.55": SafeHarborSite("Dep.55", "20", 50799100, 50801360, "intronic", "BCAS4"),
        "Prot.218": SafeHarborSite("Prot.218", "8", 119591710, 119593739, "intronic", "ENPP2"),
        "Dep.56": SafeHarborSite("Dep.56", "3", 108912857, 108913858, "intronic", "GUCA1C"),
    }

    def __init__(
        self,
        genome_path: str | Path,
        region: str | None = None,
        *,
        chromosome: str | None = None,
        position: int | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> None:
        self.genome_path = Path(genome_path)
        self._default_interval = self._resolve_interval(
            region=region,
            chromosome=chromosome,
            position=position,
            start=start,
            end=end,
            required=False,
        )

    def genomic_context(
        self,
        region: str | None = None,
        *,
        chromosome: str | None = None,
        position: int | None = None,
        start: int | None = None,
        end: int | None = None,
        target_length: int,
    ):
        """Return a callable that pads an insert to ``target_length`` with genome flanks."""

        if target_length <= 0:
            raise ValueError("target_length must be positive")

        interval = self._resolve_interval(
            region=region,
            chromosome=chromosome,
            position=position,
            start=start,
            end=end,
            required=True,
        )
        assert interval is not None
        insertion_index = interval.insertion_index

        def context_for_sequence(sequence: str | AnnotatedSequence) -> AnnotatedSequence:
            """Pad one sequence to target length with annotated genomic flanks."""

            insert = sequence if isinstance(sequence, AnnotatedSequence) else AnnotatedSequence(self._clean_sequence(sequence))
            if len(insert) > target_length:
                raise ValueError(f"Input sequence length ({len(insert)}) is longer than target_length ({target_length})")

            missing = target_length - len(insert)
            left_len = missing // 2
            right_len = missing - left_len
            left_start = insertion_index - left_len
            left = AnnotatedSequence(
                self._fetch(interval.chromosome, left_start, insertion_index),
                name="left_genome_flank",
                features=[
                    Feature(
                        "left_genome_flank",
                        0,
                        left_len,
                        type="genome_flank",
                        source="genome",
                        metadata={"chromosome": interval.chromosome, "start": left_start, "end": insertion_index},
                    )
                ],
            )
            insert = insert.add_feature("inserted_sequence", 0, len(insert), type="insert", source="manual")
            right = AnnotatedSequence(
                self._fetch(interval.chromosome, insertion_index, insertion_index + right_len),
                name="right_genome_flank",
                features=[
                    Feature(
                        "right_genome_flank",
                        0,
                        right_len,
                        type="genome_flank",
                        source="genome",
                        metadata={"chromosome": interval.chromosome, "start": insertion_index, "end": insertion_index + right_len},
                    )
                ],
            )
            return AnnotatedSequence.concat(left, insert, right, name=f"{interval.chromosome}:{insertion_index}").with_metadata(
                region=region or getattr(self._default_interval, "chromosome", None),
                insertion_chromosome=interval.chromosome,
                insertion_index=insertion_index,
            )

        return context_for_sequence

    def _resolve_interval(
        self,
        *,
        region: str | None,
        chromosome: str | None,
        position: int | None,
        start: int | None,
        end: int | None,
        required: bool,
    ) -> GenomeInterval | None:
        """Resolve named or manual coordinates into a genome interval."""

        if region is None and chromosome is None and position is None and start is None and end is None:
            default_interval = getattr(self, "_default_interval", None)
            if default_interval is not None:
                return default_interval
            if required:
                raise ValueError("Provide a safe-harbor name such as 'AAVS1', or manual coordinates.")
            return None

        if region is not None:
            if any(value is not None for value in [chromosome, position, start, end]):
                raise ValueError("Use either region lookup or manual coordinates, not both")
            site = self._safe_harbor_by_name(region)
            return GenomeInterval(site.chromosome, site.start, site.end)

        if chromosome is None:
            raise ValueError("Manual coordinates require chromosome")
        if position is not None:
            if start is not None or end is not None:
                raise ValueError("Use either position or start/end for manual coordinates")
            if position <= 0:
                raise ValueError("position must be 1-based and positive")
            return GenomeInterval(chromosome, position, position)
        if start is None or end is None:
            raise ValueError("Manual interval coordinates require both start and end")
        if start <= 0 or end <= 0 or end < start:
            raise ValueError("start/end must be positive 1-based coordinates with end >= start")
        return GenomeInterval(chromosome, start, end)

    @classmethod
    def _safe_harbor_by_name(cls, name: str) -> SafeHarborSite:
        """Find a safe-harbor site by normalized name."""

        normalized = cls._normalize_name(name)
        for key, site in cls.SAFE_HARBORS.items():
            if normalized in {cls._normalize_name(key), cls._normalize_name(site.abbrev_name)}:
                return site
        raise ValueError(f"Unknown safe-harbor abbrev name {name!r}. Available names: {', '.join(cls.SAFE_HARBORS)}")

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a safe-harbor name for lookup."""

        return re.sub(r"[^a-z0-9]+", "", name.lower())

    @staticmethod
    def _clean_sequence(sequence: str) -> str:
        """Validate and uppercase a DNA input sequence."""

        sequence = re.sub(r"\s+", "", sequence).upper()
        if not sequence:
            raise ValueError("Input sequence is empty")
        invalid = sorted(set(sequence) - set("ACGTN"))
        if invalid:
            raise ValueError(f"Input sequence contains non-DNA characters: {''.join(invalid)}")
        return sequence

    def _fetch(self, chromosome: str, start0: int, end0: int) -> str:
        """Fetch a 0-based half-open genome interval from FASTA."""

        import pysam

        with pysam.FastaFile(str(self.genome_path)) as fasta:
            reference = self._resolve_reference_name(fasta, chromosome)
            ref_len = fasta.get_reference_length(reference)
            if start0 < 0 or end0 > ref_len:
                raise ValueError(f"Requested context {reference}:{start0 + 1}-{end0} exceeds chromosome bounds 1-{ref_len}")
            return fasta.fetch(reference, start0, end0).upper()

    @staticmethod
    def _resolve_reference_name(fasta, chromosome: str) -> str:
        """Map chromosome labels to names present in the FASTA index."""

        names = set(fasta.references)
        raw = chromosome.removeprefix("chr")
        candidates = [chromosome, raw, f"chr{raw}"]
        if raw in {"M", "MT"}:
            candidates.extend(["M", "MT", "chrM", "chrMT"])
        for candidate in dict.fromkeys(candidates):
            if candidate in names:
                return candidate
        raise ValueError(f"Chromosome {chromosome!r} was not found in FASTA. Tried: {', '.join(candidates)}")
