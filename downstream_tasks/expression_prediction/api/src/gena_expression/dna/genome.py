"""Small FASTA-backed genome adapter."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

from ..sequences import AnnotatedSequence, CoordinateMap, CoordinateSegment, Feature


@dataclass(frozen=True, slots=True)
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


class _InMemoryAnnotation:
    """Small chromosome-indexed annotation backend for preloaded records."""

    def __init__(self, records_by_chrom: Mapping[str, Sequence[_GtfRecord]]) -> None:
        """Build a coordinate-sorted in-memory annotation index."""

        self.records_by_chrom = {
            str(chrom): sorted(list(records), key=lambda record: (record.start, record.end))
            for chrom, records in records_by_chrom.items()
        }

    def __bool__(self) -> bool:
        """Return whether any annotation records are available."""

        return bool(self.records_by_chrom)

    def has_chrom(self, chrom: str) -> bool:
        """Return whether the annotation has records for ``chrom``."""

        return chrom in self.records_by_chrom

    def records(self, chrom: str, start: int, end: int) -> list[_GtfRecord]:
        """Return preloaded records overlapping ``chrom:start-end``."""

        if end <= start:
            return []

        overlapping = []
        for record in self.records_by_chrom.get(chrom, []):
            if record.end <= start:
                continue
            if record.start >= end:
                break
            overlapping.append(record)
        return overlapping


class _TabixGtfAnnotation:
    """Lazy GTF/GFF reader backed by a bgzipped and tabix-indexed file."""

    def __init__(self, path: Path) -> None:
        """Store the indexed annotation path for lazy opening."""

        self.path = path
        self._tabix = None
        self._contigs: set[str] | None = None

    def __bool__(self) -> bool:
        """Return true because the configured backend is available lazily."""

        return True

    def _handle(self):
        """Open the tabix handle on first use."""

        try:
            import pysam
        except ImportError as exc:
            raise ImportError(
                "Tabix-backed GTF/GFF annotations require pysam. "
                f"debug={{'path': {str(self.path)!r}}}"
            ) from exc

        if self._tabix is None:
            try:
                self._tabix = pysam.TabixFile(str(self.path))
            except Exception as exc:
                raise OSError(
                    "Could not open tabix-indexed annotation. "
                    f"debug={{'path': {str(self.path)!r}, "
                    f"'expected_tbi': {str(self.path) + '.tbi'!r}, "
                    f"'expected_csi': {str(self.path) + '.csi'!r}}}"
                ) from exc
        return self._tabix

    def close(self) -> None:
        """Close the tabix handle, if it was opened."""

        if self._tabix is not None:
            self._tabix.close()
            self._tabix = None

    def has_chrom(self, chrom: str) -> bool:
        """Return whether the tabix index contains ``chrom``."""

        if self._contigs is None:
            self._contigs = set(self._handle().contigs)
        return chrom in self._contigs

    def records(self, chrom: str, start: int, end: int) -> list[_GtfRecord]:
        """Fetch and parse records overlapping ``chrom:start-end``."""

        if end <= start or end <= 0:
            return []
        if not self.has_chrom(chrom):
            return []

        query_start = max(0, int(start))
        query_end = max(query_start, int(end))
        if query_end <= query_start:
            return []

        try:
            rows = self._handle().fetch(chrom, query_start, query_end)
        except Exception as exc:
            raise ValueError(
                "Could not fetch annotation records from tabix index. "
                f"debug={{'path': {str(self.path)!r}, 'chrom': {chrom!r}, "
                f"'start': {start}, 'end': {end}, "
                f"'query_start': {query_start}, 'query_end': {query_end}}}"
            ) from exc

        records = []
        for raw_line in rows:
            try:
                record = Genome._parse_gtf_line(
                    raw_line,
                    path=self.path,
                    context=f"tabix query {chrom}:{query_start}-{query_end}",
                )
            except Exception as exc:
                raise ValueError(
                    "Could not parse a tabix-fetched annotation line. "
                    f"debug={{'path': {str(self.path)!r}, 'chrom': {chrom!r}, "
                    f"'start': {start}, 'end': {end}, 'line': {raw_line[:240]!r}}}"
                ) from exc
            if record is not None:
                records.append(record)
        return records


class Genome:
    """FASTA-backed genome object with chromosome-name normalization.

    ``annotation`` may be a GTF path. GTF coordinates are converted from
    1-based inclusive to internal 0-based half-open coordinates. Plain GTF/GFF
    files are bgzipped and tabix-indexed on first use; indexed files are read
    lazily by genomic interval.
    """

    def __init__(
        self,
        fasta_path: str | Path,
        annotation: str | Path | object | None = None,
        build: str | None = None,
        chrom_style: Literal["auto", "chr", "no_chr"] = "auto",
        cache: bool = True,
    ) -> None:
        """Configure FASTA access and optional genome annotations."""

        self.fasta_path = Path(fasta_path)
        self.annotation = annotation
        self.build = build
        self.chrom_style = chrom_style
        self.cache = cache
        self._fasta = None
        self._annotation = self._load_annotation(annotation) if annotation is not None else None

    def _handle(self):
        """Return a cached or one-shot FASTA handle."""

        try:
            import pysam
        except ImportError as exc:
            raise ImportError(
                "FASTA-backed genome access requires pysam. "
                f"debug={{'fasta_path': {str(self.fasta_path)!r}}}"
            ) from exc

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

        if self._annotation is None:
            return []

        normalized = self._normalize_annotation_chrom(chrom)
        records = self._annotation.records(normalized, start, end)
        type_filter = {value.lower() for value in types} if types is not None else None
        features: list[Feature] = []

        for record in records:
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
        """Configure a reusable named or explicit genomic interval."""

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

        if self._annotation is None:
            return chrom
        if self._annotation.has_chrom(chrom):
            return chrom
        raw = chrom.removeprefix("chr")
        candidates = [chrom, raw, f"chr{raw}"]
        if raw in {"M", "MT"}:
            candidates.extend(["M", "MT", "chrM", "chrMT"])
        for candidate in dict.fromkeys(candidates):
            if self._annotation.has_chrom(candidate):
                return candidate
        return chrom

    @classmethod
    def _load_annotation(cls, annotation: str | Path | object):
        """Load a supported annotation source."""

        if isinstance(annotation, (str, Path)):
            path = Path(annotation)
            if not cls._is_gtf_like_path(path):
                raise ValueError(
                    "Only GTF/GFF/GFF3 annotations are supported. "
                    f"debug={{'path': {str(path)!r}, 'suffixes': {[suffix.lower() for suffix in path.suffixes]!r}, "
                    "'supported': ['.gtf', '.gff', '.gff3', '.gtf.gz', '.gff.gz', '.gff3.gz']}}"
                )
            return cls._load_gtf(path)
        if isinstance(annotation, Mapping):
            return _InMemoryAnnotation(annotation)
        raise TypeError("annotation must be a GTF path or a chromosome-to-record mapping.")

    @classmethod
    def _load_gtf(cls, path: Path) -> _TabixGtfAnnotation:
        """Prepare a GTF/GFF file for lazy tabix-backed queries."""

        return _TabixGtfAnnotation(cls._ensure_tabix_gtf(path))

    @classmethod
    def _ensure_tabix_gtf(cls, path: Path) -> Path:
        """Return a bgzipped, tabix-indexed annotation path."""

        try:
            import pysam
        except ImportError as exc:
            raise ImportError(
                "Preparing GTF/GFF annotations for fast interval queries requires pysam. "
                f"debug={{'path': {str(path)!r}, 'suffixes': {[suffix.lower() for suffix in path.suffixes]!r}}}"
            ) from exc

        if cls._is_compressed_gtf_path(path):
            gz_path = path
        elif cls._is_plain_gtf_path(path):
            gz_path = Path(str(path) + ".gz")
            if gz_path.exists():
                warnings.warn(
                    "Plain GTF/GFF annotations are not supported directly for fast interval queries; "
                    f"using existing compressed annotation {gz_path}. "
                    f"debug={{'plain_path': {str(path)!r}, 'compressed_path': {str(gz_path)!r}}}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Plain GTF/GFF annotations are not supported directly for fast interval queries; "
                    f"creating bgzipped annotation {gz_path}. "
                    f"debug={{'plain_path': {str(path)!r}, 'compressed_path': {str(gz_path)!r}}}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                try:
                    pysam.tabix_compress(str(path), str(gz_path), force=False)
                except Exception as exc:
                    raise OSError(
                        "Could not bgzip annotation file. "
                        f"debug={{'plain_path': {str(path)!r}, 'compressed_path': {str(gz_path)!r}, "
                        f"'exists_plain': {path.exists()}, 'exists_compressed': {gz_path.exists()}}}"
                    ) from exc
        else:
            raise ValueError(
                "Only GTF/GFF/GFF3 annotations are supported. "
                f"debug={{'path': {str(path)!r}, 'suffixes': {[suffix.lower() for suffix in path.suffixes]!r}}}"
            )

        tbi_path = Path(str(gz_path) + ".tbi")
        csi_path = Path(str(gz_path) + ".csi")
        if not tbi_path.exists() and not csi_path.exists():
            sorted_gz_path = cls._sorted_gtf_path(gz_path)
            if sorted_gz_path.exists():
                return cls._sort_and_index_gtf(
                    gz_path,
                    pysam,
                    original_error=RuntimeError("original annotation has no tabix index and a sorted copy already exists"),
                )
            warnings.warn(
                f"Creating tabix index for annotation {gz_path}. "
                f"debug={{'compressed_path': {str(gz_path)!r}, 'tbi_path': {str(tbi_path)!r}, "
                f"'csi_path': {str(csi_path)!r}}}",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                pysam.tabix_index(str(gz_path), preset="gff", force=False)
            except Exception as exc:
                return cls._sort_and_index_gtf(gz_path, pysam, original_error=exc)

        return gz_path

    @classmethod
    def _sort_and_index_gtf(cls, gz_path: Path, pysam, *, original_error: Exception) -> Path:
        """Sort an unindexed annotation with Polars, bgzip it, and tabix-index it."""

        sorted_gz_path = cls._sorted_gtf_path(gz_path)
        if sorted_gz_path.exists():
            warnings.warn(
                "Could not index the provided annotation; using existing sorted annotation. "
                f"debug={{'compressed_path': {str(gz_path)!r}, 'sorted_path': {str(sorted_gz_path)!r}, "
                f"'original_error': {str(original_error)!r}}}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "Could not index the provided annotation, likely because it is unsorted; "
                "sorting with Polars and writing a sorted bgzipped copy. "
                f"debug={{'compressed_path': {str(gz_path)!r}, 'sorted_path': {str(sorted_gz_path)!r}, "
                f"'original_error': {str(original_error)!r}}}",
                RuntimeWarning,
                stacklevel=2,
            )
            cls._write_sorted_gtf(gz_path, sorted_gz_path, pysam)

        sorted_tbi_path = Path(str(sorted_gz_path) + ".tbi")
        sorted_csi_path = Path(str(sorted_gz_path) + ".csi")
        if not sorted_tbi_path.exists() and not sorted_csi_path.exists():
            try:
                pysam.tabix_index(str(sorted_gz_path), preset="gff", force=False)
            except Exception as exc:
                raise OSError(
                    "Could not create tabix index for sorted annotation. The original file may have invalid "
                    "GTF/GFF rows, unsupported coordinates, or chromosome ordering that tabix cannot index. "
                    f"debug={{'compressed_path': {str(gz_path)!r}, 'sorted_path': {str(sorted_gz_path)!r}, "
                    f"'sorted_tbi_path': {str(sorted_tbi_path)!r}, 'sorted_csi_path': {str(sorted_csi_path)!r}, "
                    f"'exists_sorted': {sorted_gz_path.exists()}, 'exists_sorted_tbi': {sorted_tbi_path.exists()}, "
                    f"'exists_sorted_csi': {sorted_csi_path.exists()}, "
                    f"'original_index_error': {str(original_error)!r}, 'sorted_index_error': {str(exc)!r}}}"
                ) from exc

        return sorted_gz_path

    @classmethod
    def _write_sorted_gtf(cls, gz_path: Path, sorted_gz_path: Path, pysam) -> None:
        """Sort GTF/GFF rows by chromosome and coordinates using Polars."""

        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "Sorting an unindexed GTF/GFF annotation requires polars. "
                f"debug={{'compressed_path': {str(gz_path)!r}, 'sorted_path': {str(sorted_gz_path)!r}}}"
            ) from exc

        columns = ["chrom", "source", "feature_type", "start", "end", "score", "strand", "frame", "attributes"]
        plain_sorted_path = cls._plain_path_for_compressed_gtf(sorted_gz_path)
        try:
            df = pl.read_csv(
                str(gz_path),
                separator="\t",
                has_header=False,
                comment_prefix="#",
                new_columns=columns,
                quote_char=None,
            )
            if df.width != len(columns):
                raise ValueError(
                    "Expected 9 columns while reading GTF/GFF annotation with Polars. "
                    f"debug={{'compressed_path': {str(gz_path)!r}, 'observed_columns': {df.width}, "
                    f"'expected_columns': {len(columns)}}}"
                )
            df = df.with_columns(
                pl.col("start").cast(pl.Int64),
                pl.col("end").cast(pl.Int64),
            ).sort(["chrom", "start", "end"])
            df.write_csv(str(plain_sorted_path), separator="\t", include_header=False)
            pysam.tabix_compress(str(plain_sorted_path), str(sorted_gz_path), force=False)
        except Exception as exc:
            raise OSError(
                "Could not sort and bgzip annotation. "
                f"debug={{'compressed_path': {str(gz_path)!r}, 'plain_sorted_path': {str(plain_sorted_path)!r}, "
                f"'sorted_path': {str(sorted_gz_path)!r}, 'exists_input': {gz_path.exists()}, "
                f"'exists_plain_sorted': {plain_sorted_path.exists()}, 'exists_sorted': {sorted_gz_path.exists()}, "
                f"'error': {str(exc)!r}}}"
            ) from exc
        finally:
            if plain_sorted_path.exists():
                plain_sorted_path.unlink()

    @staticmethod
    def _sorted_gtf_path(path: Path) -> Path:
        """Return the sibling sorted annotation path for a compressed GTF/GFF."""

        lower_name = path.name.lower()
        for suffix in (".gtf.gz", ".gff.gz", ".gff3.gz"):
            if lower_name.endswith(suffix):
                return path.with_name(path.name[: -len(suffix)] + f".sorted{suffix}")
        return path.with_name(path.name + ".sorted.gtf.gz")

    @staticmethod
    def _plain_path_for_compressed_gtf(path: Path) -> Path:
        """Return the temporary plain path used before bgzip compression."""

        if path.name.lower().endswith(".gz"):
            return path.with_name(path.name[:-3] + ".tmp")
        return path.with_suffix(path.suffix + ".plain")

    @staticmethod
    def _is_plain_gtf_path(path: Path) -> bool:
        """Return whether ``path`` looks like a plain GTF/GFF file."""

        return path.suffix.lower() in {".gtf", ".gff", ".gff3"}

    @staticmethod
    def _is_compressed_gtf_path(path: Path) -> bool:
        """Return whether ``path`` looks like a compressed GTF/GFF file."""

        return path.name.lower().endswith((".gtf.gz", ".gff.gz", ".gff3.gz"))

    @classmethod
    def _is_gtf_like_path(cls, path: Path) -> bool:
        """Return whether ``path`` has a supported annotation extension."""

        return cls._is_plain_gtf_path(path) or cls._is_compressed_gtf_path(path)

    @classmethod
    def _parse_gtf_line(
        cls,
        raw_line: str,
        *,
        path: Path | None = None,
        line_number: int | None = None,
        context: str | None = None,
    ) -> _GtfRecord | None:
        """Parse one GTF/GFF row into internal coordinates."""

        line = raw_line.rstrip("\n")
        if not line or line.startswith("#"):
            return None

        fields = line.split("\t", 8)
        if len(fields) != 9:
            raise ValueError(
                "Invalid GTF/GFF row: expected 9 tab-separated columns. "
                f"debug={{'path': {str(path) if path is not None else None!r}, "
                f"'line_number': {line_number}, 'context': {context!r}, "
                f"'observed_columns': {len(fields)}, 'line': {line[:240]!r}}}"
            )

        chrom, source, feature_type, start_text, end_text, score, strand, frame, attrs_text = fields
        try:
            start0 = int(start_text) - 1
            end0 = int(end_text)
        except ValueError as exc:
            raise ValueError(
                "Invalid GTF/GFF coordinates: start and end must be integers. "
                f"debug={{'path': {str(path) if path is not None else None!r}, "
                f"'line_number': {line_number}, 'context': {context!r}, "
                f"'chrom': {chrom!r}, 'start_text': {start_text!r}, 'end_text': {end_text!r}, "
                f"'line': {line[:240]!r}}}"
            ) from exc
        if start0 < 0 or end0 < start0:
            raise ValueError(
                "Invalid GTF/GFF coordinates after conversion to 0-based half-open coordinates. "
                f"debug={{'path': {str(path) if path is not None else None!r}, "
                f"'line_number': {line_number}, 'context': {context!r}, "
                f"'chrom': {chrom!r}, 'start_text': {start_text!r}, 'end_text': {end_text!r}, "
                f"'start0': {start0}, 'end0': {end0}, 'line': {line[:240]!r}}}"
            )

        return _GtfRecord(
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

    @staticmethod
    def _parse_gtf_attributes(text: str) -> dict[str, str]:
        """Parse the ninth GTF column into a simple string dictionary."""

        attributes: dict[str, str] = {}
        for chunk in text.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk and (" " not in chunk or chunk.index("=") < chunk.index(" ")):
                key, value = chunk.split("=", 1)
            else:
                parts = chunk.split(None, 1)
                if len(parts) != 2:
                    continue
                key, value = parts
            attributes[key] = value.strip().strip('"')
        return attributes

    def chrom_length(self, chrom: str) -> int:
        """Return chromosome length from the FASTA index."""

        return self._handle().get_reference_length(self.normalize_chrom(chrom))

    def close(self) -> None:
        """Close cached FASTA and annotation handles, if open."""

        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None
        if self._annotation is not None and hasattr(self._annotation, "close"):
            self._annotation.close()


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
        """Configure a reusable named or explicit genomic interval."""

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
