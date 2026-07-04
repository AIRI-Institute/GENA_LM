"""Annotation-aware plasmid context helpers."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .sequences import AnnotatedSequence, Feature


PrimerTailEntry = tuple[str, str] | Mapping[str, object]
PrimerTails = Mapping[str, PrimerTailEntry]


@dataclass(frozen=True)
class FeatureMatch:
    """Reporter feature match parsed from a GenBank record."""

    key: str
    start: int  # 1-based inclusive, GenBank-style
    end: int  # 1-based inclusive, GenBank-style
    strand: int
    note: str


@dataclass(frozen=True)
class PlasmidMetadata:
    """Minimal metadata needed to construct a plasmid from NCBI."""

    name: str
    ncbi_link: str


class PlasmidRecord:
    """GenBank/FASTA plasmid record with annotations converted to features."""

    def __init__(
        self,
        sequence: str,
        *,
        name: str | None = None,
        genbank_text: str | None = None,
        reporter_feature_name: str = "luc2",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.sequence = self._clean_sequence(sequence)
        self.name = name or "plasmid"
        self.genbank_text = genbank_text or ""
        self.reporter_feature_name = reporter_feature_name
        self.metadata = dict(metadata or {})
        self._feature_records = self._parse_feature_records(self.genbank_text) if self.genbank_text else []
        self.features = self._parse_features()
        self.reporter_feature = self._find_reporter_feature(reporter_feature_name) if self._feature_records else None

    @classmethod
    def from_files(
        cls,
        genbank_path: str | Path,
        fasta_path: str | Path | None = None,
        *,
        reporter_feature_name: str = "luc2",
    ) -> "PlasmidRecord":
        """Load a plasmid from local GenBank and optional FASTA files."""

        genbank_path = Path(genbank_path)
        genbank_text = genbank_path.read_text(encoding="utf-8")
        sequence = cls._read_fasta_sequence(Path(fasta_path)) if fasta_path else cls._read_origin_sequence(genbank_text)
        name = cls._parse_locus_name(genbank_text) or genbank_path.stem
        return cls(
            sequence,
            name=name,
            genbank_text=genbank_text,
            reporter_feature_name=reporter_feature_name,
            metadata={"genbank_path": str(genbank_path), "fasta_path": str(fasta_path) if fasta_path else None},
        )

    @classmethod
    def from_ncbi(
        cls,
        ncbi_url: str,
        data_storage: str | Path,
        *,
        reporter_feature_name: str = "luc2",
    ) -> "PlasmidRecord":
        """Download and cache GenBank/FASTA records from NCBI."""

        accession = cls._accession_from_url(ncbi_url)
        plasmid_dir = Path(data_storage) / accession
        plasmid_dir.mkdir(parents=True, exist_ok=True)
        genbank_path = plasmid_dir / f"{accession}.gb"
        fasta_path = plasmid_dir / f"{accession}.fa"

        if not genbank_path.exists():
            genbank_path.write_text(cls._download_text(cls._ncbi_sviewer_url(accession, "genbank")), encoding="utf-8")
        if not fasta_path.exists():
            fasta_path.write_text(cls._download_text(cls._ncbi_sviewer_url(accession, "fasta")), encoding="utf-8")

        return cls.from_files(genbank_path, fasta_path, reporter_feature_name=reporter_feature_name)

    @staticmethod
    def _clean_sequence(sequence: str) -> str:
        """Normalize a DNA sequence."""

        sequence = re.sub(r"\s+", "", sequence).upper()
        invalid = sorted(set(sequence) - set("ACGTN"))
        if invalid:
            raise ValueError(f"Plasmid sequence contains non-DNA characters: {''.join(invalid)}")
        return sequence

    @staticmethod
    def _accession_from_url(ncbi_url: str) -> str:
        """Extract the NCBI accession from a nuccore URL."""

        parsed = urllib.parse.urlparse(ncbi_url)
        accession = parsed.path.rstrip("/").split("/")[-1]
        if not accession:
            raise ValueError(f"Could not parse accession from URL: {ncbi_url}")
        return accession

    @staticmethod
    def _download_text(url: str) -> str:
        """Download a text record from NCBI."""

        req = urllib.request.Request(url, headers={"User-Agent": "variant-api-plasmid-context/1.0"})
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")

    @staticmethod
    def _ncbi_sviewer_url(accession: str, report: str) -> str:
        query = urllib.parse.urlencode({"id": accession, "db": "nuccore", "report": report, "retmode": "text"})
        return f"https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi?{query}"

    @staticmethod
    def _read_fasta_sequence(path: Path) -> str:
        """Read a FASTA file into a single sequence."""

        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(">"):
                lines.append(line.strip())
        return "".join(lines).upper()

    @staticmethod
    def _read_origin_sequence(genbank_text: str) -> str:
        """Read sequence bases from the ORIGIN block of a GenBank flatfile."""

        match = re.search(r"^ORIGIN\s*(.*?)^//", genbank_text, flags=re.MULTILINE | re.DOTALL)
        if not match:
            raise ValueError("GenBank record does not contain an ORIGIN block; pass fasta_path explicitly.")
        return re.sub(r"[^A-Za-z]", "", match.group(1)).upper()

    @staticmethod
    def _parse_locus_name(genbank_text: str) -> str | None:
        """Parse the LOCUS name from a GenBank flatfile."""

        match = re.search(r"^LOCUS\s+(\S+)", genbank_text, flags=re.MULTILINE)
        return match.group(1) if match else None

    @staticmethod
    def _safe_key(text: str) -> str:
        """Normalize feature labels for dictionary keys."""

        key = re.sub(r"\s+", " ", text.strip())
        return key if key else "unnamed_feature"

    @staticmethod
    def _parse_location(location: str) -> tuple[int, int, int]:
        """Parse a simple GenBank location into 1-based start/end and strand."""

        strand = -1 if location.startswith("complement(") else 1
        numbers = [int(number) for number in re.findall(r"\d+", location)]
        if not numbers:
            raise ValueError(f"Could not parse feature location: {location}")
        return min(numbers), max(numbers), strand

    @staticmethod
    def _parse_qualifiers(block: str) -> dict[str, str]:
        """Parse quoted GenBank feature qualifiers from a feature block."""

        qualifiers: dict[str, str] = {}
        matches = list(re.finditer(r'/(\w+)="', block))
        for index, match in enumerate(matches):
            key = match.group(1)
            value_start = match.end()
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            value = block[value_start:value_end]
            value = value.rsplit('"', 1)[0]
            qualifiers[key] = re.sub(r"\s+", " ", value).strip()
        return qualifiers

    def _parse_feature_records(self, genbank_text: str) -> list[dict[str, object]]:
        """Parse GenBank FEATURES into searchable records."""

        in_features = False
        current: dict[str, object] | None = None
        records: list[dict[str, object]] = []

        for raw_line in genbank_text.splitlines():
            if raw_line.startswith("FEATURES"):
                in_features = True
                continue
            if in_features and raw_line.startswith("ORIGIN"):
                break
            if not in_features:
                continue

            match = re.match(r"^\s{5}(\S+)\s+(.+)$", raw_line)
            if match:
                if current:
                    records.append(current)
                feature_type, location = match.groups()
                current = {"type": feature_type, "location": location.strip(), "block": raw_line}
            elif current:
                current["block"] = str(current["block"]) + "\n" + raw_line

        if current:
            records.append(current)

        for record in records:
            start, end, strand = self._parse_location(str(record["location"]))
            qualifiers = self._parse_qualifiers(str(record["block"]))
            label = (
                qualifiers.get("gene")
                or qualifiers.get("product")
                or qualifiers.get("note")
                or qualifiers.get("label")
                or str(record["type"])
            )
            record.update(
                {
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "qualifiers": qualifiers,
                    "label": self._safe_key(label),
                    "search_text": self._safe_key(
                        " ".join([str(record["type"]), str(record["location"]), str(record["block"])])
                    ).lower(),
                }
            )
        return records

    def _parse_features(self) -> dict[str, tuple[int, int]]:
        """Expose parsed feature spans by label in GenBank coordinates."""

        features: dict[str, tuple[int, int]] = {}
        counts: dict[str, int] = {}
        for feature in self._feature_records:
            label = str(feature["label"])
            counts[label] = counts.get(label, 0) + 1
            key = label if counts[label] == 1 else f"{label}#{counts[label]}"
            features[key] = (int(feature["start"]), int(feature["end"]))
        return features

    def _find_reporter_feature(self, reporter_feature_name: str) -> FeatureMatch:
        """Find the longest feature containing the reporter substring."""

        needle = reporter_feature_name.lower()
        matches = []
        for feature in self._feature_records:
            if needle in str(feature["search_text"]):
                qualifiers = feature.get("qualifiers", {})
                matches.append(
                    FeatureMatch(
                        key=str(feature["label"]),
                        start=int(feature["start"]),
                        end=int(feature["end"]),
                        strand=int(feature["strand"]),
                        note="; ".join(str(value) for value in dict(qualifiers).values()),
                    )
                )

        if not matches:
            available = "\n".join(f"- {key}: {value}" for key, value in self.features.items())
            raise ValueError(
                f"Reporter feature containing {reporter_feature_name!r} was not found.\n"
                f"Available parsed features:\n{available}"
            )
        return max(matches, key=lambda match: match.end - match.start + 1)

    def reporter_tss_index(self) -> int:
        """Return reporter TSS as 0-based sequence index."""

        if self.reporter_feature is None:
            raise ValueError("Reporter feature is not available for this plasmid.")
        return self.reporter_feature.start - 1 if self.reporter_feature.strand >= 0 else self.reporter_feature.end - 1

    def annotated_sequence(self) -> AnnotatedSequence:
        """Return the plasmid sequence with parsed GenBank features attached."""

        features = []
        for record in self._feature_records:
            strand = "+" if int(record["strand"]) >= 0 else "-"
            qualifiers = dict(record.get("qualifiers", {}))
            features.append(
                Feature(
                    name=str(record["label"]),
                    start=int(record["start"]) - 1,
                    end=int(record["end"]),
                    type=str(record["type"]),
                    strand=strand,
                    source="genbank",
                    metadata={
                        "location": record["location"],
                        "qualifiers": qualifiers,
                        "genbank_start": int(record["start"]),
                        "genbank_end": int(record["end"]),
                    },
                )
            )
        return AnnotatedSequence(
            self.sequence,
            name=self.name,
            features=features,
            metadata={**self.metadata, "record_type": "plasmid"},
        )

    def plasmid_context(
        self,
        *,
        context_size: int | None = None,
        circular: bool = True,
        allow_repeats: bool = False,
        primer_tails: PrimerTails,
    ) -> Callable[[str | AnnotatedSequence, str], AnnotatedSequence]:
        """Return an annotation-aware MPRA fragment context function."""

        if context_size is not None and context_size <= 0:
            raise ValueError("context_size must be positive")

        base = self.annotated_sequence()
        old_tss = self.reporter_tss_index()
        old_reporter_anchor = self.sequence[old_tss : old_tss + 40]
        insert_fns: dict[str, Callable[[AnnotatedSequence, str], AnnotatedSequence]] = {}

        def get_insert_fn(element: str) -> Callable[[AnnotatedSequence, str], AnnotatedSequence]:
            element_key = self._resolve_element_key(element, primer_tails)
            if element_key not in insert_fns:
                left_tail, reverse_primer_tail, primer_match = _primer_tail_entry_parts(primer_tails[element_key])
                insert_fns[element_key] = make_annotated_mcs_insert_fn(
                    left_tail,
                    reverse_primer_tail,
                    primer_match=primer_match,
                    primer_element=element_key,
                )
            return insert_fns[element_key]

        def context_for_mpra(mpra_fragment: str | AnnotatedSequence, element: str, *args, **kwargs) -> AnnotatedSequence:
            """Insert one MPRA fragment and return its annotated plasmid context."""

            edited = get_insert_fn(element)(base, mpra_fragment).with_metadata(element=element)
            new_tss = edited.sequence.find(old_reporter_anchor)
            if new_tss == -1:
                new_tss = old_tss + (len(edited) - len(base))

            target_size = len(edited) if context_size is None else context_size
            fill_sequence = base if not allow_repeats and target_size > len(edited) else None
            context = centered_plasmid_window(
                edited,
                center=new_tss,
                size=target_size,
                circular=circular,
                fill_sequence=fill_sequence,
                fill_center=old_tss,
            )
            tss = len(context) // 2
            debug = dict(context.metadata.get("plasmid_context_debug", {}))
            debug.update(
                {
                    "requested_element": element,
                    "resolved_element": self._resolve_element_key(element, primer_tails),
                    "context_size": context_size,
                    "target_size": target_size,
                    "final_context_length": len(context),
                    "old_reporter_tss": old_tss,
                    "new_reporter_tss": new_tss,
                    "reporter_anchor_found": edited.sequence.find(old_reporter_anchor) != -1,
                    "reporter_tss_in_context": tss,
                    "allow_repeats": allow_repeats,
                    "used_fill_sequence": fill_sequence is not None,
                }
            )
            return context.add_feature(
                "reporter_tss",
                tss,
                tss + 1,
                type="tss",
                source="plasmid",
                metadata={"reporter_feature_name": self.reporter_feature_name, "element": element},
            ).with_metadata(plasmid_context_debug=debug)

        return context_for_mpra

    @staticmethod
    def _normalize_element_name(element: str) -> str:
        """Normalize an element label for tolerant lookup."""

        return re.sub(r"[^a-z0-9]+", "", element.lower())

    @classmethod
    def _resolve_element_key(cls, element: str, primer_tails: PrimerTails) -> str:
        """Map a dataset element label to a primer-tail table key."""

        candidates = [element]
        for separator in (".", "-"):
            if separator in element:
                candidates.append(element.split(separator, 1)[0])

        normalized_to_key = {cls._normalize_element_name(key): key for key in primer_tails}
        for candidate in candidates:
            normalized = cls._normalize_element_name(candidate)
            if normalized in normalized_to_key:
                return normalized_to_key[normalized]

        normalized_element = cls._normalize_element_name(element)
        prefix_matches = [
            key
            for key in primer_tails
            if normalized_element.startswith(cls._normalize_element_name(key))
            or cls._normalize_element_name(key).startswith(normalized_element)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        if prefix_matches:
            raise ValueError(f"Element name {element!r} matched multiple primer-tail elements: {', '.join(prefix_matches)}")
        raise ValueError(f"Unknown MPRA element {element!r}. Available primer-tail elements: {', '.join(primer_tails)}")


class PlasmidCollection:
    """Route MPRA elements to source plasmids and cache annotated contexts.

    This is the annotation-aware replacement for the old
    ``PlasmidCollection(...).element_name_based_context`` workflow. It accepts
    your ``CONST.enhancer_to_mpra_plasmid`` and
    ``CONST.promoter_to_mpra_plasmid`` merged mapping as ``element_to_plasmid``.
    """

    def __init__(
        self,
        data_storage: str | Path,
        *,
        context_size: int | None = None,
        circular: bool = True,
        allow_repeats: bool = False,
        primer_tails: PrimerTails,
        reporter_feature_name: str = "luc2",
        element_to_plasmid: Mapping[str, Any],
    ) -> None:
        self.data_storage = Path(data_storage)
        self.context_size = context_size
        self.circular = circular
        self.allow_repeats = allow_repeats
        self.primer_tails = primer_tails
        self.reporter_feature_name = reporter_feature_name
        self.element_to_plasmid = dict(element_to_plasmid)
        self._plasmids: dict[str, PlasmidRecord] = {}
        self._context_fns: dict[str, Callable[[str | AnnotatedSequence, str], AnnotatedSequence]] = {}

    def element_name_based_context(
        self,
        mpra_fragment: str | AnnotatedSequence,
        element_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> AnnotatedSequence:
        """Return an annotated MPRA fragment in the plasmid context for its element."""

        metadata, canonical_element = self._resolve_element_metadata(element_name)
        context_fn = self._context_fn_for_plasmid(metadata)
        return context_fn(mpra_fragment, canonical_element, *args, **kwargs)

    def _context_fn_for_plasmid(
        self,
        metadata: Any,
    ) -> Callable[[str | AnnotatedSequence, str], AnnotatedSequence]:
        """Get or create the cached context function for one plasmid."""

        plasmid_name = str(getattr(metadata, "name"))
        if plasmid_name not in self._context_fns:
            plasmid = self._plasmids.get(plasmid_name)
            if plasmid is None:
                plasmid = PlasmidRecord.from_ncbi(
                    str(getattr(metadata, "ncbi_link")),
                    data_storage=self.data_storage,
                    reporter_feature_name=self.reporter_feature_name,
                )
                self._plasmids[plasmid_name] = plasmid

            self._context_fns[plasmid_name] = plasmid.plasmid_context(
                context_size=self.context_size,
                circular=self.circular,
                allow_repeats=self.allow_repeats,
                primer_tails=self.primer_tails,
            )

        return self._context_fns[plasmid_name]

    def _resolve_element_metadata(self, element_name: str) -> tuple[Any, str]:
        """Resolve an element label to plasmid metadata and canonical element name."""

        normalized_query = self._normalize_element_name(element_name)
        normalized_to_key = {self._normalize_element_name(key): key for key in self.element_to_plasmid}

        candidate_keys = [element_name]
        for separator in (".", "-"):
            if separator in element_name:
                candidate_keys.append(element_name.split(separator, 1)[0])

        for candidate in candidate_keys:
            normalized = self._normalize_element_name(candidate)
            if normalized in normalized_to_key:
                key = normalized_to_key[normalized]
                return self.element_to_plasmid[key], self._primer_tail_element_name(key)

        prefix_matches = [
            key
            for key in self.element_to_plasmid
            if self._normalize_element_name(key).startswith(normalized_query)
            or normalized_query.startswith(self._normalize_element_name(key))
        ]
        if len(prefix_matches) == 1:
            key = prefix_matches[0]
            return self.element_to_plasmid[key], self._primer_tail_element_name(key)
        if prefix_matches:
            raise ValueError(f"Element name {element_name!r} matched multiple plasmids: {', '.join(prefix_matches)}")

        raise ValueError(f"Unknown MPRA element {element_name!r}. Available elements: {', '.join(self.element_to_plasmid)}")

    def _primer_tail_element_name(self, element_name: str) -> str:
        """Resolve the element name used by the primer-tail table."""

        return PlasmidRecord._resolve_element_key(element_name, self.primer_tails)

    @staticmethod
    def _normalize_element_name(element_name: str) -> str:
        """Normalize an element name for dictionary matching."""

        return re.sub(r"[^a-z0-9]+", "", element_name.lower())


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""

    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(table)[::-1].upper()


def _best_primer_suffix_tail(
    primer: str,
    target: str,
    expected_position: int,
    min_match: int = 12,
    edge_slop: int = 25,
) -> tuple[str, str, int]:
    """Infer the non-annealing primer tail from suffix matches."""

    primer = primer.upper()
    target = target.upper()
    candidates = []

    for match_len in range(len(primer), min_match - 1, -1):
        annealing = primer[-match_len:]
        start = 0
        while True:
            position = target.find(annealing, start)
            if position == -1:
                break
            distance = abs(position - expected_position)
            candidates.append((distance, -match_len, primer[:-match_len], annealing, position))
            start = position + 1

    near_edge = [candidate for candidate in candidates if candidate[0] <= edge_slop]
    if near_edge:
        _, _, tail, annealing, position = min(near_edge)
        return tail, annealing, position

    if candidates:
        _, _, tail, annealing, position = min(candidates)
        return tail, annealing, position

    raise ValueError(
        f"Could not infer cloning tail for primer {primer!r}; "
        f"no suffix of at least {min_match} bp matched the genomic target."
    )


def _match_primer_pair(
    forward_primer: str,
    reverse_primer: str,
    sequence: str,
    min_match: int,
    expected_position: int,
    edge_slop: int,
) -> dict[str, object]:
    """Infer primer tails and orientation for one construct sequence."""

    candidates = []
    orientations = {
        "plus": (sequence, reverse_complement(sequence)),
        "minus": (reverse_complement(sequence), sequence),
    }

    for orientation, (forward_target, reverse_target) in orientations.items():
        try:
            forward_tail, forward_annealing, forward_position = _best_primer_suffix_tail(
                forward_primer,
                forward_target,
                expected_position=expected_position,
                min_match=min_match,
                edge_slop=edge_slop,
            )
            reverse_tail, reverse_annealing, reverse_position = _best_primer_suffix_tail(
                reverse_primer,
                reverse_target,
                expected_position=expected_position,
                min_match=min_match,
                edge_slop=edge_slop,
            )
        except ValueError:
            continue

        edge_distance = abs(forward_position - expected_position) + abs(reverse_position - expected_position)
        candidates.append(
            {
                "orientation": orientation,
                "forward_tail": forward_tail,
                "reverse_tail": reverse_tail,
                "forward_annealing": forward_annealing,
                "reverse_annealing": reverse_annealing,
                "forward_match_len": len(forward_annealing),
                "reverse_match_len": len(reverse_annealing),
                "forward_match_position": forward_position,
                "reverse_match_position": reverse_position,
                "edge_distance": edge_distance,
            }
        )

    if not candidates:
        raise ValueError("Could not infer primer tails in either plus or minus genomic orientation.")

    return max(
        candidates,
        key=lambda item: (
            -int(item["edge_distance"]),
            min(int(item["forward_match_len"]), int(item["reverse_match_len"])),
            int(item["forward_match_len"]) + int(item["reverse_match_len"]),
        ),
    )


def _fetch_hg38_interval(
    fasta,
    chromosome: str,
    start: int,
    end: int,
    flank: int = 0,
) -> str:
    """Fetch a GRCh38 interval with optional flanking sequence."""

    names = set(fasta.references)
    candidates = [chromosome, f"chr{chromosome}"]
    if chromosome == "MT":
        candidates.extend(["M", "chrM"])

    for name in candidates:
        if name in names:
            fetch_start = max(0, start - 1 - flank)
            fetch_end = end + flank
            return fasta.fetch(name, fetch_start, fetch_end).upper()

    raise ValueError(f"Chromosome {chromosome!r} was not found in FASTA. Tried: {', '.join(candidates)}")


def _primer_tail_entry_parts(entry: PrimerTailEntry) -> tuple[str, str, dict[str, object] | None]:
    """Return cloning tails and optional primer-match details from one entry."""

    if isinstance(entry, Mapping):
        left_tail = str(entry["forward_tail"])
        reverse_primer_tail = str(entry["reverse_tail"])
        return left_tail, reverse_primer_tail, dict(entry)
    left_tail, reverse_primer_tail = entry
    return str(left_tail), str(reverse_primer_tail), None


def infer_table18_primer_tails(
    fasta_reference_path: str | Path,
    *,
    table18_primers: Mapping[str, tuple[str, str]],
    locations: Mapping[str, tuple[str, int, int]],
    flank: int = 100,
    min_match: int = 12,
    edge_slop: int = 25,
    return_details: bool = False,
) -> PrimerTails:
    """Infer cloning tails from a Supplementary-Table-18-style primer table.

    The API does not hard-code CAGI constants. Pass your ``CONST.table18_primers``
    and ``CONST.locations`` mappings explicitly.
    """

    import pysam

    tails: dict[str, tuple[str, str] | dict[str, object]] = {}
    with pysam.FastaFile(str(fasta_reference_path)) as fasta:
        for element, (forward_primer, reverse_primer) in table18_primers.items():
            chromosome, start, end = locations[element]
            sequence = _fetch_hg38_interval(fasta, chromosome, start, end, flank=flank)
            match = _match_primer_pair(
                forward_primer,
                reverse_primer,
                sequence,
                min_match=min_match,
                expected_position=flank,
                edge_slop=edge_slop,
            )
            if return_details:
                tails[element] = match
            else:
                tails[element] = (str(match["forward_tail"]), str(match["reverse_tail"]))

    return tails  # type: ignore[return-value]


def _assembled_insert(
    left_tail: str,
    mpra_fragment: str | AnnotatedSequence,
    reverse_primer_tail: str,
) -> AnnotatedSequence:
    """Build annotated ``left scar + MPRA fragment + right scar``."""

    left_tail = left_tail.upper()
    fragment = (
        mpra_fragment
        if isinstance(mpra_fragment, AnnotatedSequence)
        else AnnotatedSequence(str(mpra_fragment).upper(), name="mpra_fragment")
    )
    right_scar = reverse_complement(reverse_primer_tail)
    left = AnnotatedSequence(left_tail, name="left_primer_tail").add_feature(
        "left_mcs_scar", 0, len(left_tail), type="scar", source="plasmid"
    )
    right = AnnotatedSequence(right_scar, name="right_primer_tail").add_feature(
        "right_mcs_scar", 0, len(right_scar), type="scar", source="plasmid"
    )
    fragment = fragment.add_feature("mpra_insert", 0, len(fragment), type="insert", source="mpra")
    return AnnotatedSequence.concat(left, fragment, right, name="assembled_mpra_insert")


def make_annotated_mcs_insert_fn(
    left_tail: str,
    reverse_primer_tail: str,
    *,
    primer_match: Mapping[str, object] | None = None,
    primer_element: str | None = None,
) -> Callable[[AnnotatedSequence, str | AnnotatedSequence], AnnotatedSequence]:
    """Build an annotation-preserving MCS insertion function."""

    left_tail = left_tail.upper()
    right_scar = reverse_complement(reverse_primer_tail)

    def insert_into_mcs(plasmid: AnnotatedSequence, mpra_fragment: str | AnnotatedSequence) -> AnnotatedSequence:
        """Replace MCS scars with an annotated MPRA insert."""

        plasmid_sequence = plasmid.sequence.upper()
        replacement = _assembled_insert(left_tail, mpra_fragment, reverse_primer_tail)
        n = len(plasmid_sequence)
        circular_sequence = plasmid_sequence + plasmid_sequence
        left = circular_sequence.find(left_tail, 0, n + len(left_tail) - 1)
        if left == -1:
            raise ValueError("Could not find left MCS scar in circular plasmid.")

        right = circular_sequence.find(right_scar, left, left + n + len(right_scar) - 1)
        if right == -1:
            raise ValueError("Could not find right MCS scar after left scar in circular plasmid.")

        right_end = right + len(right_scar)
        left_end = left + len(left_tail)
        overlapped_features = _overlapped_features(plasmid, left, right_end)
        if right < left_end:
            warnings.warn(
                "MCS scars overlap in the plasmid sequence; replacing the full span from left scar start to right scar end.",
                RuntimeWarning,
                stacklevel=2,
            )

        if right_end <= n:
            edited = plasmid.replace(left, right_end, replacement, preserve_partial_features=True)
        else:
            right_end_mod = right_end % n
            remaining = plasmid[right_end_mod:left]
            assert isinstance(remaining, AnnotatedSequence)
            edited = AnnotatedSequence.concat(replacement, remaining, name=plasmid.name)

        debug = {
            "plasmid_name": plasmid.name,
            "plasmid_length": n,
            "primer_element": primer_element,
            "left_tail": left_tail,
            "right_scar": right_scar,
            "left_tail_length": len(left_tail),
            "reverse_primer_tail_length": len(reverse_primer_tail),
            "right_scar_length": len(right_scar),
            "left_tail_position": left,
            "left_tail_end": left_end,
            "right_scar_position_unwrapped": right,
            "right_scar_position": right % n,
            "right_scar_end_unwrapped": right_end,
            "right_scar_end": right_end % n,
            "replacement_span_length": right_end - left,
            "replacement_crosses_origin": right_end > n,
            "inserted_length": len(replacement),
            "mpra_fragment_length": len(mpra_fragment) if isinstance(mpra_fragment, AnnotatedSequence) else len(str(mpra_fragment)),
            "overlapped_features": overlapped_features,
            "preserved_overlapping_features": [
                feature.to_dict()
                for feature in edited.features
                if feature.metadata and feature.metadata.get("clipped_by_replacement")
            ],
            "inserted_features": [feature.to_dict() for feature in replacement.features],
            "primer_match_details_available": primer_match is not None,
            "primer_match": _primer_match_debug(primer_match),
        }
        return edited.with_metadata(
            mcs_left=left,
            mcs_right=right % n,
            mcs_right_end=right_end % n,
            inserted_length=len(replacement),
            plasmid_context_debug=debug,
        )

    return insert_into_mcs


def _primer_match_debug(primer_match: Mapping[str, object] | None) -> dict[str, object] | None:
    """Return a compact, JSON-friendly primer matching debug payload."""

    if primer_match is None:
        return None
    keys = [
        "orientation",
        "forward_tail",
        "reverse_tail",
        "forward_annealing",
        "reverse_annealing",
        "forward_match_len",
        "reverse_match_len",
        "forward_match_position",
        "reverse_match_position",
        "edge_distance",
    ]
    return {key: primer_match[key] for key in keys if key in primer_match}


def _overlapped_features(
    plasmid: AnnotatedSequence,
    start: int,
    end: int,
) -> list[dict[str, object]]:
    """Return plasmid features overlapping an unwrapped replacement interval."""

    n = len(plasmid)
    if n <= 0:
        return []

    features: list[dict[str, object]] = []
    first_copy = start // n - 1
    last_copy = end // n + 2
    for copy_index in range(first_copy, last_copy):
        offset = copy_index * n
        for feature in plasmid.features:
            feature_start = feature.start + offset
            feature_end = feature.end + offset
            overlap_start = max(feature_start, start)
            overlap_end = min(feature_end, end)
            if overlap_end <= overlap_start:
                continue
            features.append(
                {
                    "name": feature.name,
                    "type": feature.type,
                    "strand": feature.strand,
                    "source": feature.source,
                    "feature_start": feature.start,
                    "feature_end": feature.end,
                    "source_copy": copy_index,
                    "overlap_start_unwrapped": overlap_start,
                    "overlap_end_unwrapped": overlap_end,
                    "overlap_start": overlap_start % n,
                    "overlap_end": overlap_end % n,
                    "metadata": dict(feature.metadata or {}),
                }
            )
    return features


def centered_plasmid_window(
    sequence: AnnotatedSequence,
    *,
    center: int,
    size: int,
    circular: bool,
    fill_sequence: AnnotatedSequence | None = None,
    fill_center: int | None = None,
) -> AnnotatedSequence:
    """Return an annotated window centered on ``center``.

    When ``fill_sequence`` is provided, bases outside the single edited plasmid
    copy are filled from the original plasmid, matching the attached context
    helper's no-repeat behavior.
    """

    half_left = size // 2
    window_start = center - half_left
    window_end = window_start + size

    if not circular:
        start = max(0, window_start)
        end = min(len(sequence), start + size)
        window = sequence[start:end]
        assert isinstance(window, AnnotatedSequence)
        return window.with_metadata(window_center=center, window_start=start)

    if fill_sequence is None:
        return _circular_window(sequence, window_start, window_end).with_metadata(
            window_center=center,
            window_start=window_start,
            circular=True,
        )

    if fill_center is None:
        fill_center = center

    out = []
    for out_i in range(size):
        offset = out_i - half_left
        edited_i = center + offset
        if 0 <= edited_i < len(sequence):
            out.append(sequence.sequence[edited_i])
        else:
            fill_i = (fill_center + offset) % len(fill_sequence)
            out.append(fill_sequence.sequence[fill_i])

    features: list[Feature] = []
    features.extend(_linear_window_features(sequence, window_start, window_end, role="edited_plasmid"))
    fill_window_start = fill_center - half_left
    fill_window = _circular_window(fill_sequence, fill_window_start, fill_window_start + size)
    for interval_start, interval_end in _outside_edited_intervals(window_start, size, len(sequence)):
        for feature in fill_window.features:
            clipped = feature.clip(interval_start, interval_end)
            if clipped is not None:
                metadata = {**dict(clipped.metadata or {}), "window_role": "fill_plasmid"}
                features.append(
                    Feature(
                        clipped.name,
                        clipped.start + interval_start,
                        clipped.end + interval_start,
                        type=clipped.type,
                        strand=clipped.strand,
                        source=clipped.source,
                        metadata=metadata,
                    )
                )

    return AnnotatedSequence(
        "".join(out),
        name=sequence.name,
        features=features,
        metadata={**dict(sequence.metadata), "window_center": center, "window_start": window_start, "circular": True},
    )


def _outside_edited_intervals(window_start: int, size: int, edited_length: int) -> list[tuple[int, int]]:
    """Return local output intervals filled from the original plasmid."""

    intervals = []
    left_end = min(size, max(0, -window_start))
    if left_end > 0:
        intervals.append((0, left_end))
    right_start = max(0, edited_length - window_start)
    if right_start < size:
        intervals.append((right_start, size))
    return intervals


def _linear_window_features(
    sequence: AnnotatedSequence,
    window_start: int,
    window_end: int,
    *,
    role: str,
) -> list[Feature]:
    """Clip non-circular feature coordinates to one linear window."""

    features = []
    for feature in sequence.features:
        clipped = feature.clip(window_start, window_end)
        if clipped is None:
            continue
        metadata = {**dict(clipped.metadata or {}), "window_role": role}
        features.append(
            Feature(
                clipped.name,
                clipped.start,
                clipped.end,
                type=clipped.type,
                strand=clipped.strand,
                source=clipped.source,
                metadata=metadata,
            )
        )
    return features


def _circular_window(
    sequence: AnnotatedSequence,
    window_start: int,
    window_end: int,
) -> AnnotatedSequence:
    """Slice an annotated circular sequence over an unwrapped coordinate range."""

    n = len(sequence)
    out = "".join(sequence.sequence[i % n] for i in range(window_start, window_end))
    features: list[Feature] = []
    first_copy = window_start // n - 1
    last_copy = window_end // n + 2
    for copy_index in range(first_copy, last_copy):
        offset = copy_index * n
        for feature in sequence.features:
            shifted_start = feature.start + offset
            shifted_end = feature.end + offset
            clipped_start = max(shifted_start, window_start)
            clipped_end = min(shifted_end, window_end)
            if clipped_end <= clipped_start:
                continue
            metadata = {**dict(feature.metadata or {}), "source_copy": copy_index, "window_role": "circular_plasmid"}
            features.append(
                Feature(
                    feature.name,
                    clipped_start - window_start,
                    clipped_end - window_start,
                    type=feature.type,
                    strand=feature.strand,
                    source=feature.source,
                    metadata=metadata,
                )
            )
    return AnnotatedSequence(out, name=sequence.name, features=features, metadata=sequence.metadata)
