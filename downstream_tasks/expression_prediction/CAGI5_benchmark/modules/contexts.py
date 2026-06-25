from __future__ import annotations

import re
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple, ClassVar


InsertFn = Callable[[str, str], str]
PrimerTails = Mapping[str, Tuple[str, str]]



table18_primers: Dict[str, Tuple[str, str]] = {
    "F9": ("CCCGGGCTCGAGATCTCCCACTGATGAACTGTGC", "CCGGATTGCCAAGCTTAACCTTTGCTAGCAGATTGTG"),
    "FOXE1": ("CCCGGGCTCGAGATCTCTCGCCAGCGGTCCGCAGG", "CCGGATTGCCAAGCTTGGCCTGGCGTCCCCGGAACG"),
    "GP1BA": ("CCCGGGCTCGAGATCTTTGTGAATGCCGCGTCCTG", "CCGGATTGCCAAGCTTACGACCAGAGCTCCTCTC"),
    "HBB": ("CCCGGGCTCGAGATCTAAGGACAGGTACGGCTGTC", "CCGGATTGCCAAGCTTGGTGTCTGTTTGAGGTTGC"),
    "HBG1": ("CCCGGGCTCGAGATCTGCAGTATCCTCTTGGGGGCC", "CCGGATTGCCAAGCTTGGCGTCTGGACTAGGAGCTTATTG"),
    "HNF4A": ("CCCGGGCTCGAGATCCCCCAGAGTGCAGGACTAG", "CCGGATTGCCAAGCTGGCCAAGCCCACCCAG"),
    "LDLR": ("CCCGGGCTCGAGATCTAGCTCTTCACCGGAGACCCA", "CCGGATTGCCAAGCTTGCTCGCAGCCTCTGCCAG"),
    "MSMB": ("CCCGGGCTCGAGATCAAAGGTCCAGCAATTCAGC", "CCGGATTGCCAAGCTAAGCAGGACTCCTTATAGACAGG"),
    "PKLR": ("CCCGGGCTCGAGATCTAGGTTACAGAGTGGTGAAGGC", "CCGGATTGCCAAGCTTGCTTTCAGTGTGGGCCTGG"),
    "TERT": ("CCCGGGCTCGAGATCCCAGGACCGCGCTTCCCAC", "CCGGATTGCCAAGCTCGCGGGGGTGGCCGGG"),
    "SORT1": ("GAGGATATCAAGATCTGAACTGGAAAAGCCCTGTCCGG", "TCTAGTGTCTAAGCTTCAGACCCCCGGGACTGGAC"),
    "IRF4": ("GAGGATATCAAGATCTGGCGTGTCCGCCTGTTGG", "TCTAGTGTCTAAGCTTACGGGGGTAAAGGAGTGC"),
    "IRF6": ("GAGGATATCAAGATCTTCTGTTTGCTTAGCTTACCTC", "TCTAGTGTCTAAGCTTGTAAATGGTGAGTAGGAAGTTG"),
    "MYC (rs6983267)": ("GAGGATATCAAGATCTCTGCATCGCTCCATAGAG", "TCTAGTGTCTAAGCTTTGCTGGTAGAACTTACG"),
    "MYC (rs11986220)": ("GAGGATATCAAGATCTGGTAAGTCAACATGAAATTATAAACC", "TCTAGTGTCTAAGCTTCAAGTACTGTGGGGGTTTTGTTAG"),
    "TCF7L2": ("GAGGATATCAAGATCTAGGTTCTGTTTCTTGCTTAG", "TCTAGTGTCTAAGCTTATTACAAATTATTAGAACTTTC"),
    "ZFAND3": ("GAGGATATCAAGATCTTTCATGTTTCCCCCGTATGTG", "TCTAGTGTCTAAGCTTTCCTGCCCCAAGTTGCACAGC"),
    "BCL11A": ("GCTCGCTAGCCTCGAGCCTAACACAGTAGCTGGTACCTG", "CGCCGAGGCCAGATCTGTACTGATGGACCTTGGGTG"),
    "UC88": ("GAGGATATCAAGATCTTACAGATAAATGCACACATGTATACG", "TCTAGTGTCTAAGCTTGGGACTCGGTGGCGGTG"),
    "ZRS": ("TGGCCTAACTGGCCGGTACCTGAGATATGGCTTCATTTTCTGT", "ATGATCTAAGCTTAAGGCTGAGCAACATGACAGCAC"),
    "RET": ("CTAGCCCGGGCTCGAGCAGAGGCACCAGGGTCAAAGC", "TGCAGATCGCAGATCTGAAGCCCAGAATTCCCGCTGC"),
}


promotors_locations: Dict[str, Tuple[int, int]] = {
    "F9": (139530463, 139530765),
    "FOXE1": (97853255, 97853854),
    "GP1BA": (19723266, 19723650),
    "HBB": (5227022, 5227208),
    "HBG1": (5249805, 5250078),
    "HNF4A": (44355520, 44355804),
    "LDLR": (11089231, 11089548),
    "MSMB": (46046244, 46046834),
    "PKLR": (155301395, 155301864),
    "TERT": (1294989, 1295247),
}


promoter_chromosomes: Dict[str, str] = {
    "F9": "X",
    "FOXE1": "9",
    "GP1BA": "22",
    "HBB": "11",
    "HBG1": "11",
    "HNF4A": "20",
    "LDLR": "19",
    "MSMB": "10",
    "PKLR": "1",
    "TERT": "5",
}


enchancer_locations: Dict[str, Tuple[int, int]] = {
    "BCL11A": (60494940, 60495539),
    "IRF4": (396143, 396593),
    "IRF6": (209815790, 209816390),
    "MYC (rs6983267)": (127400829, 127401428),
    "MYC (rs11986220)": (127519270, 127519732),
    "RET": (43086479, 43087078),
    "SORT1": (109274652, 109275251),
    "TCF7L2": (112998240, 112998839),
    "UC88": (161238408, 161238997),
    "ZFAND3": (37807499, 37808077),
    "ZRS": (156791119, 156791603),
}


enhancer_chromosomes: Dict[str, str] = {
    "BCL11A": "2",
    "IRF4": "6",
    "IRF6": "1",
    "MYC (rs6983267)": "8",
    "MYC (rs11986220)": "8",
    "RET": "10",
    "SORT1": "1",
    "TCF7L2": "10",
    "UC88": "2",
    "ZFAND3": "6",
    "ZRS": "7",
}

@dataclass(frozen=True)
class PlasmidMetadata:
    name: str
    ncbi_link: str


PLASMIDS = {
    "pGL4.11b": PlasmidMetadata(
        name="pGL4.11b",
        ncbi_link="https://www.ncbi.nlm.nih.gov/nuccore/MK484103.1",
    ),
    "pGL4.11c": PlasmidMetadata(
        name="pGL4.11c",
        ncbi_link="https://www.ncbi.nlm.nih.gov/nuccore/MK484104.1",
    ),
    "pGL4.23c": PlasmidMetadata(
        name="pGL4.23c",
        ncbi_link="https://www.ncbi.nlm.nih.gov/nuccore/MK484105.1",
    ),
    "pGL4.23d": PlasmidMetadata(
        name="pGL4.23d",
        ncbi_link="https://www.ncbi.nlm.nih.gov/nuccore/MK484106.1",
    ),
    "pGL3c": PlasmidMetadata(
        name="pGL3c",
        ncbi_link="https://www.ncbi.nlm.nih.gov/nuccore/MK484107.1",
    ),
    "pGL4Zc": PlasmidMetadata(
        name="pGL4Zc",
        ncbi_link="https://www.ncbi.nlm.nih.gov/nuccore/MK484108.1",
    ),
}


enhancer_to_mpra_plasmid = {
    "BCL11A": PLASMIDS["pGL4.23d"],
    "IRF4": PLASMIDS["pGL4.23c"], #changed from d
    "IRF6": PLASMIDS["pGL4.23c"],
    "MYCrs6983267": PLASMIDS["pGL4.23c"],
    "MYCrs11986220": PLASMIDS["pGL4.23c"], #changed from d
    "RET": PLASMIDS["pGL3c"],
    "SORT1": PLASMIDS["pGL4.23c"],
    "TCF7L2": PLASMIDS["pGL4.23c"], #changed from d
    "UC88": PLASMIDS["pGL4.23c"],
    "ZFAND3": PLASMIDS["pGL4.23c"],
    "ZRS": PLASMIDS["pGL4Zc"],
}


promoter_to_mpra_plasmid = {
    "F9": PLASMIDS["pGL4.11c"],
    "FOXE1": PLASMIDS["pGL4.11c"],
    "GP1BB": PLASMIDS["pGL4.11c"],
    "HBB": PLASMIDS["pGL4.11c"],
    "HBG1": PLASMIDS["pGL4.11c"],
    "HNF4A": PLASMIDS["pGL4.11c"],
    "LDLR": PLASMIDS["pGL4.11b"],
    "MSMB": PLASMIDS["pGL4.11c"],
    "PKLR": PLASMIDS["pGL4.11c"],
    "TERT": PLASMIDS["pGL4.11b"],
}


@dataclass(frozen=True)
class FeatureMatch:
    key: str
    start: int          # 1-based inclusive, GenBank-style
    end: int            # 1-based inclusive, GenBank-style
    strand: int
    note: str


class Plasmid:
    """
    Download and hold a GenBank/FASTA plasmid record from NCBI.

    Coordinates exposed in `features` are 1-based inclusive, matching GenBank.
    Internally, reporter/TSS centering uses Biopython's 0-based half-open feature
    coordinates.
    """

    def __init__(
        self,
        ncbi_url: str,
        data_storage: str | Path,
        reporter_feature_name: str = "luc2",
    ):
        """Load a plasmid record and identify the reporter feature for TSS anchoring."""
        self.ncbi_url = ncbi_url
        self.accession = self._accession_from_url(ncbi_url)
        self.data_storage = Path(data_storage)
        self.reporter_feature_name = reporter_feature_name

        self.plasmid_dir = self.data_storage / self.accession
        self.plasmid_dir.mkdir(parents=True, exist_ok=True)

        self.genbank_path = self.plasmid_dir / f"{self.accession}.gb"
        self.fasta_path = self.plasmid_dir / f"{self.accession}.fa"

        self._download_if_needed()
        self.genbank_text = self.genbank_path.read_text(encoding="utf-8")
        self.sequence = self._read_fasta_sequence(self.fasta_path)
        self.plasmid_name = self._parse_locus_name(self.genbank_text) or self.accession

        self._feature_records = self._parse_feature_records(self.genbank_text)
        self.features: Dict[str, Tuple[int, int]] = self._parse_features()
        self.reporter_feature = self._find_reporter_feature(reporter_feature_name)

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
        req = urllib.request.Request(url, headers={"User-Agent": "plasmid-context/1.0"})
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")

    def _ncbi_sviewer_url(self, report: str) -> str:
        """Build an NCBI sviewer URL for the requested report type."""
        query = urllib.parse.urlencode(
            {
                "id": self.accession,
                "db": "nuccore",
                "report": report,
                "retmode": "text",
            }
        )
        return f"https://www.ncbi.nlm.nih.gov/sviewer/viewer.cgi?{query}"

    def _download_if_needed(self) -> None:
        """Download GenBank and FASTA files when they are not cached locally."""
        if not self.genbank_path.exists():
            self.genbank_path.write_text(
                self._download_text(self._ncbi_sviewer_url("genbank")),
                encoding="utf-8",
            )

        if not self.fasta_path.exists():
            fasta = self._download_text(self._ncbi_sviewer_url("fasta"))
            self.fasta_path.write_text(fasta, encoding="utf-8")

    @staticmethod
    def _safe_key(text: str) -> str:
        """Normalize feature labels for dictionary keys."""
        key = re.sub(r"\s+", " ", text.strip())
        return key if key else "unnamed_feature"

    @staticmethod
    def _read_fasta_sequence(path: Path) -> str:
        """Read a FASTA file into a single uppercase sequence."""
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(">"):
                lines.append(line.strip())
        return "".join(lines).upper()

    @staticmethod
    def _parse_locus_name(genbank_text: str) -> Optional[str]:
        """Parse the LOCUS name from a GenBank flatfile."""
        match = re.search(r"^LOCUS\s+(\S+)", genbank_text, flags=re.MULTILINE)
        return match.group(1) if match else None

    @staticmethod
    def _parse_location(location: str) -> Tuple[int, int, int]:
        """Parse a simple GenBank location into start, end, and strand."""
        strand = -1 if location.startswith("complement(") else 1
        numbers = [int(number) for number in re.findall(r"\d+", location)]
        if not numbers:
            raise ValueError(f"Could not parse feature location: {location}")
        return min(numbers), max(numbers), strand

    @staticmethod
    def _parse_qualifiers(block: str) -> Dict[str, str]:
        """Parse quoted GenBank feature qualifiers from a feature block."""
        qualifiers: Dict[str, str] = {}
        matches = list(re.finditer(r'/(\w+)="', block))
        for index, match in enumerate(matches):
            key = match.group(1)
            value_start = match.end()
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            value = block[value_start:value_end]
            value = value.rsplit('"', 1)[0]
            qualifiers[key] = re.sub(r"\s+", " ", value).strip()
        return qualifiers

    def _parse_feature_records(self, genbank_text: str):
        """Parse GenBank FEATURES into searchable feature records."""
        in_features = False
        current = None
        records = []

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
                current["block"] += "\n" + raw_line

        if current:
            records.append(current)

        for record in records:
            start, end, strand = self._parse_location(record["location"])
            qualifiers = self._parse_qualifiers(record["block"])
            label = (
                qualifiers.get("gene")
                or qualifiers.get("product")
                or qualifiers.get("note")
                or qualifiers.get("label")
                or record["type"]
            )
            record.update(
                {
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "qualifiers": qualifiers,
                    "label": self._safe_key(label),
                    "search_text": self._safe_key(
                        " ".join([record["type"], record["location"], record["block"]])
                    ).lower(),
                }
            )

        return records

    def _parse_features(self) -> Dict[str, Tuple[int, int]]:
        """Expose parsed feature spans by label."""
        features: Dict[str, Tuple[int, int]] = {}
        counts: Dict[str, int] = {}

        for feature in self._feature_records:
            label = feature["label"]
            counts[label] = counts.get(label, 0) + 1
            key = label if counts[label] == 1 else f"{label}#{counts[label]}"
            features[key] = (feature["start"], feature["end"])

        return features

    def _find_reporter_feature(self, reporter_feature_name: str) -> FeatureMatch:
        """Find the longest feature containing the reporter substring."""
        needle = reporter_feature_name.lower()
        matches = []

        for feature in self._feature_records:
            if needle in feature["search_text"]:
                matches.append(
                    FeatureMatch(
                        key=feature["label"],
                        start=feature["start"],
                        end=feature["end"],
                        strand=feature["strand"],
                        note="; ".join(feature["qualifiers"].values()),
                    )
                )

        if not matches:
            available = "\n".join(f"- {k}: {v}" for k, v in self.features.items())
            raise ValueError(
                f"Reporter feature containing {reporter_feature_name!r} was not found.\n"
                f"Available parsed features:\n{available}"
            )

        return max(matches, key=lambda match: match.end - match.start + 1)

    def reporter_tss_index(self) -> int:
        """Return reporter TSS as 0-based sequence index."""
        feature = self.reporter_feature
        return feature.start - 1 if feature.strand >= 0 else feature.end - 1

    def plasmid_context(
        self,
        context_size: int | None = None,
        fasta_reference_path: str | Path | None = None,
        circular: bool = True,
        allow_repeats: bool = False,
        primer_tails: PrimerTails | None = None,
        tail_inference_kwargs: Optional[Dict[str, object]] = None,
    ) -> Callable[[str, str], str]:
        """
        Return a callable that inserts an MPRA fragment with element-specific scars.

        The returned callable expects at least:
            context_for_mpra(mpra_fragment, element)

        If `context_size=None`, the returned sequence is the full edited plasmid,
        circularly rotated so the reporter TSS is near the middle. No padding is
        added in that mode.

        Extra positional/keyword arguments, such as cell type from a dataset map,
        are accepted and ignored by this plasmid-context step.
        """

        if context_size is not None and context_size <= 0:
            raise ValueError("context_size must be positive")
        if primer_tails is None and fasta_reference_path is None:
            raise ValueError(
                "Provide fasta_reference_path so primer tails can be inferred, "
                "or pass precomputed primer_tails."
            )

        old_tss = self.reporter_tss_index()
        old_reporter_anchor = self.sequence[old_tss : old_tss + 40]
        inferred_tails: PrimerTails | None = primer_tails
        insert_fns: Dict[str, InsertFn] = {}

        def get_primer_tails() -> PrimerTails:
            """Load or reuse inferred element-specific primer tails."""
            nonlocal inferred_tails
            if inferred_tails is None:
                kwargs = tail_inference_kwargs or {}
                inferred_tails = infer_table18_primer_tails(
                    fasta_reference_path,
                    **kwargs,
                )
            return inferred_tails

        def get_insert_fn(element: str) -> InsertFn:
            """Build or reuse an insertion function for one MPRA element."""
            tails = get_primer_tails()
            element_key = self._resolve_element_key(element, tails)
            if element_key not in insert_fns:
                left_tail, reverse_primer_tail = tails[element_key]
                insert_fns[element_key] = make_mcs_insert_fn(
                    left_tail,
                    reverse_primer_tail,
                )
            return insert_fns[element_key]

        def context_for_mpra(
            mpra_fragment: str,
            element: str,
            *args,
            **kwargs,
        ) -> str:
            """Insert one MPRA fragment and return its plasmid context."""
            insert_fn = get_insert_fn(element)
            edited = insert_fn(self.sequence, mpra_fragment.upper()).upper()
            new_tss = edited.find(old_reporter_anchor)

            if new_tss == -1:
                new_tss = old_tss + (len(edited) - len(self.sequence))

            target_size = len(edited) if context_size is None else context_size
            fill_sequence = (
                self.sequence
                if not allow_repeats and target_size > len(edited)
                else None
            )

            return self._centered_window(
                sequence=edited,
                center=new_tss,
                size=target_size,
                circular=circular,
                fill_sequence=fill_sequence,
                fill_center=old_tss,
            )

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

        normalized_to_key = {
            cls._normalize_element_name(key): key
            for key in primer_tails
        }
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
            raise ValueError(
                f"Element name {element!r} matched multiple primer-tail elements: "
                + ", ".join(prefix_matches)
            )

        available = ", ".join(primer_tails)
        raise ValueError(
            f"Unknown MPRA element {element!r}. Available primer-tail elements: {available}"
        )

    @staticmethod
    def _centered_window(
        sequence: str,
        center: int,
        size: int,
        circular: bool,
        fill_sequence: str | None = None,
        fill_center: int | None = None,
    ) -> str:
        """Return a centered sequence window, optionally filling from backbone DNA."""
        half_left = size // 2

        if circular:
            # Old behavior: repeat the whole edited plasmid, including insert.
            if fill_sequence is None:
                n = len(sequence)
                start = center - half_left
                return "".join(sequence[i % n] for i in range(start, start + size))

            # New behavior: use the edited plasmid once.
            # Outside that one edited copy, fill from original plasmid DNA.
            if fill_center is None:
                fill_center = center

            fill_sequence = fill_sequence.upper()
            sequence = sequence.upper()

            out = []
            fill_n = len(fill_sequence)

            for out_i in range(size):
                offset = out_i - half_left
                edited_i = center + offset

                if 0 <= edited_i < len(sequence):
                    out.append(sequence[edited_i])
                else:
                    fill_i = (fill_center + offset) % fill_n
                    out.append(fill_sequence[fill_i])

            return "".join(out)

        start = max(0, center - half_left)
        end = min(len(sequence), start + size)
        return sequence[start:end]


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
) -> Tuple[str, str, int]:
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
):
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
        raise ValueError(
            "Could not infer primer tails in either plus or minus genomic orientation."
        )

    return max(
        candidates,
        key=lambda item: (
            -item["edge_distance"],
            min(item["forward_match_len"], item["reverse_match_len"]),
            item["forward_match_len"] + item["reverse_match_len"],
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

    raise ValueError(
        f"Chromosome {chromosome!r} was not found in FASTA. "
        f"Tried: {', '.join(candidates)}"
    )


def infer_table18_primer_tails(
    fasta_reference_path: str | Path,
    flank: int = 100,
    min_match: int = 12,
    edge_slop: int = 25,
    return_details: bool = False,
) -> Dict[str, Tuple[str, str]] | Dict[str, Dict[str, str | int]]:
    """
    Infer the cloning tails in Supplementary Table 18 from a hg38 reference FASTA.

    Returns by default:
        {
            element: (forward_tail, reverse_tail),
            ...
        }

    If `return_details=True`, returns match diagnostics including matched
    annealing sequence, matched length, and position in the flanked search
    window. Matches shorter than about 12 bp are usually not trustworthy.

    Coordinates are the lower-row GRCh38 coordinates from Supplementary Tables 1-2.
    `flank` extends each hg38 interval before searching, because several primer
    annealing sites sit just outside or not exactly at the tabulated construct
    coordinate edge.

    `edge_slop` prefers primer matches within this many bp of the expected
    construct edge in the flanked window.

    The reverse primer tail is returned in primer orientation, not reverse-complemented
    plasmid-scar orientation.
    """
    import pysam

    primers = {
        "F9": ("CCCGGGCTCGAGATCTCCCACTGATGAACTGTGC", "CCGGATTGCCAAGCTTAACCTTTGCTAGCAGATTGTG"),
        "FOXE1": ("CCCGGGCTCGAGATCTCTCGCCAGCGGTCCGCAGG", "CCGGATTGCCAAGCTTGGCCTGGCGTCCCCGGAACG"),
        "GP1BA": ("CCCGGGCTCGAGATCTTTGTGAATGCCGCGTCCTG", "CCGGATTGCCAAGCTTACGACCAGAGCTCCTCTC"),
        "HBB": ("CCCGGGCTCGAGATCTAAGGACAGGTACGGCTGTC", "CCGGATTGCCAAGCTTGGTGTCTGTTTGAGGTTGC"),
        "HBG1": ("CCCGGGCTCGAGATCTGCAGTATCCTCTTGGGGGCC", "CCGGATTGCCAAGCTTGGCGTCTGGACTAGGAGCTTATTG"),
        "HNF4A": ("CCCGGGCTCGAGATCCCCCAGAGTGCAGGACTAG", "CCGGATTGCCAAGCTGGCCAAGCCCACCCAG"),
        "LDLR": ("CCCGGGCTCGAGATCTAGCTCTTCACCGGAGACCCA", "CCGGATTGCCAAGCTTGCTCGCAGCCTCTGCCAG"),
        "MSMB": ("CCCGGGCTCGAGATCAAAGGTCCAGCAATTCAGC", "CCGGATTGCCAAGCTAAGCAGGACTCCTTATAGACAGG"),
        "PKLR": ("CCCGGGCTCGAGATCTAGGTTACAGAGTGGTGAAGGC", "CCGGATTGCCAAGCTTGCTTTCAGTGTGGGCCTGG"),
        "TERT": ("CCCGGGCTCGAGATCCCAGGACCGCGCTTCCCAC", "CCGGATTGCCAAGCTCGCGGGGGTGGCCGGG"),
        "SORT1": ("GAGGATATCAAGATCTGAACTGGAAAAGCCCTGTCCGG", "TCTAGTGTCTAAGCTTCAGACCCCCGGGACTGGAC"),
        "IRF4": ("GAGGATATCAAGATCTGGCGTGTCCGCCTGTTGG", "TCTAGTGTCTAAGCTTACGGGGGTAAAGGAGTGC"),
        "IRF6": ("GAGGATATCAAGATCTTCTGTTTGCTTAGCTTACCTC", "TCTAGTGTCTAAGCTTGTAAATGGTGAGTAGGAAGTTG"),
        "MYC (rs6983267)": ("GAGGATATCAAGATCTCTGCATCGCTCCATAGAG", "TCTAGTGTCTAAGCTTTGCTGGTAGAACTTACG"),
        "MYC (rs11986220)": ("GAGGATATCAAGATCTGGTAAGTCAACATGAAATTATAAACC", "TCTAGTGTCTAAGCTTCAAGTACTGTGGGGGTTTTGTTAG"),
        "TCF7L2": ("GAGGATATCAAGATCTAGGTTCTGTTTCTTGCTTAG", "TCTAGTGTCTAAGCTTATTACAAATTATTAGAACTTTC"),
        "ZFAND3": ("GAGGATATCAAGATCTTTCATGTTTCCCCCGTATGTG", "TCTAGTGTCTAAGCTTTCCTGCCCCAAGTTGCACAGC"),
        "BCL11A": ("GCTCGCTAGCCTCGAGCCTAACACAGTAGCTGGTACCTG", "CGCCGAGGCCAGATCTGTACTGATGGACCTTGGGTG"),
        "UC88": ("GAGGATATCAAGATCTTACAGATAAATGCACACATGTATACG", "TCTAGTGTCTAAGCTTGGGACTCGGTGGCGGTG"),
        "ZRS": ("TGGCCTAACTGGCCGGTACCTGAGATATGGCTTCATTTTCTGT", "ATGATCTAAGCTTAAGGCTGAGCAACATGACAGCAC"),
        "RET": ("CTAGCCCGGGCTCGAGCAGAGGCACCAGGGTCAAAGC", "TGCAGATCGCAGATCTGAAGCCCAGAATTCCCGCTGC"),
    }
    locations = {
        "F9": ("X", 139530463, 139530765),
        "FOXE1": ("9", 97853255, 97853854),
        "GP1BA": ("22", 19723266, 19723650),
        "HBB": ("11", 5227022, 5227208),
        "HBG1": ("11", 5249805, 5250078),
        "HNF4A": ("20", 44355520, 44355804),
        "LDLR": ("19", 11089231, 11089548),
        "MSMB": ("10", 46046244, 46046834),
        "PKLR": ("1", 155301395, 155301864),
        "TERT": ("5", 1294989, 1295247),
        "BCL11A": ("2", 60494940, 60495539),
        "IRF4": ("6", 396143, 396593),
        "IRF6": ("1", 209815790, 209816390),
        "MYC (rs6983267)": ("8", 127400829, 127401428),
        "MYC (rs11986220)": ("8", 127519270, 127519732),
        "RET": ("10", 43086479, 43087078),
        "SORT1": ("1", 109274652, 109275251),
        "TCF7L2": ("10", 112998240, 112998839),
        "UC88": ("2", 161238408, 161238997),
        "ZFAND3": ("6", 37807499, 37808077),
        "ZRS": ("7", 156791119, 156791603),
    }

    tails: Dict[str, Tuple[str, str]] | Dict[str, Dict[str, str | int]] = {}
    with pysam.FastaFile(str(fasta_reference_path)) as fasta:
        for element, (forward_primer, reverse_primer) in primers.items():
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
                tails[element] = (match["forward_tail"], match["reverse_tail"])

    return tails


def make_mcs_insert_fn(left_tail: str, reverse_primer_tail: str) -> InsertFn:
    """
    Build an insertion function using scars inferred from Supplementary Table 18.

    Example:
        left_tail = "CCCGGGCTCGAGATCT"
        reverse_primer_tail = "CCGGATTGCCAAGCTT"
    """
    left_tail = left_tail.upper()
    right_scar = reverse_complement(reverse_primer_tail)

    def insert_into_mcs(plasmid_sequence: str, mpra_fragment: str) -> str:
        """Replace the MCS region with scars plus the MPRA fragment."""
        plasmid_sequence = plasmid_sequence.upper()
        mpra_fragment = mpra_fragment.upper()
        assembled_insert = left_tail + mpra_fragment + right_scar

        n = len(plasmid_sequence)
        circular_sequence = plasmid_sequence + plasmid_sequence
        left = circular_sequence.find(left_tail, 0, n + len(left_tail) - 1)
        if left == -1:
            raise ValueError(
                "Could not find left MCS scar in circular plasmid. Check tails, "
                "orientation, or whether this exact plasmid contains the MCS."
            )

        right_search_start = left
        right_search_stop = left + n
        right = circular_sequence.find(
            right_scar,
            right_search_start,
            right_search_stop + len(right_scar) - 1,
        )
        if right == -1:
            raise ValueError(
                "Could not find right MCS scar after left scar in circular plasmid. "
                "Check tails, orientation, or whether this exact plasmid contains the MCS."
            )

        right_end = right + len(right_scar)
        left_end = left + len(left_tail)
        if right < left_end:
            warnings.warn(
                "MCS scars overlap in the plasmid sequence; replacing the full "
                "span from left scar start to right scar end.",
                RuntimeWarning,
                stacklevel=2,
            )

        if right_end <= n:
            return (
                plasmid_sequence[:left]
                + assembled_insert
                + plasmid_sequence[right_end:]
            )

        right_end_mod = right_end % n
        return assembled_insert + plasmid_sequence[right_end_mod:left]

    return insert_into_mcs


class PlasmidCollection:
    """
    Route MPRA elements to their source plasmids and cache initialized contexts.

    The method `element_name_based_context(...)` follows the callable contract used
    by dataset transforms: sequence first, element second, optional extra args
    after that.
    """

    def __init__(
        self,
        data_storage: str | Path,
        *,
        context_size: int | None = None,
        fasta_reference_path: str | Path | None = None,
        circular: bool = True,
        allow_repeats: bool = False,
        primer_tails: PrimerTails | None = None,
        tail_inference_kwargs: Optional[Dict[str, object]] = None,
        reporter_feature_name: str = "luc2",
        element_to_plasmid: Mapping[str, PlasmidMetadata] | None = None,
    ):
        """Create a cached element-to-plasmid context router."""
        self.data_storage = Path(data_storage)
        self.context_size = context_size
        self.fasta_reference_path = fasta_reference_path
        self.circular = circular
        self.allow_repeats = allow_repeats
        self.primer_tails = primer_tails
        self.tail_inference_kwargs = tail_inference_kwargs
        self.reporter_feature_name = reporter_feature_name
        self.element_to_plasmid = dict(element_to_plasmid or self._default_element_to_plasmid())
        self._plasmids: Dict[str, Plasmid] = {}
        self._context_fns: Dict[str, Callable[[str, str], str]] = {}

    @staticmethod
    def _default_element_to_plasmid() -> Dict[str, PlasmidMetadata]:
        """Merge the default enhancer and promoter plasmid maps."""
        return {
            **enhancer_to_mpra_plasmid,
            **promoter_to_mpra_plasmid,
        }

    def element_name_based_context(
        self,
        mpra_fragment: str,
        element_name: str,
        *args,
        **kwargs,
    ) -> str:
        """Return an MPRA fragment in the plasmid context for its element."""
        metadata, canonical_element = self._resolve_element_metadata(element_name)
        context_fn = self._context_fn_for_plasmid(metadata)
        return context_fn(mpra_fragment, canonical_element, *args, **kwargs)

    def _context_fn_for_plasmid(
        self,
        metadata: PlasmidMetadata,
    ) -> Callable[[str, str], str]:
        """Get or create the cached context function for one plasmid."""
        if metadata.name not in self._context_fns:
            plasmid = self._plasmids.get(metadata.name)
            if plasmid is None:
                plasmid = Plasmid(
                    metadata.ncbi_link,
                    data_storage=self.data_storage,
                    reporter_feature_name=self.reporter_feature_name,
                )
                self._plasmids[metadata.name] = plasmid

            self._context_fns[metadata.name] = plasmid.plasmid_context(
                context_size=self.context_size,
                fasta_reference_path=self.fasta_reference_path,
                circular=self.circular,
                allow_repeats=self.allow_repeats,
                primer_tails=self.primer_tails,
                tail_inference_kwargs=self.tail_inference_kwargs,
            )

        return self._context_fns[metadata.name]

    def _resolve_element_metadata(self, element_name: str) -> Tuple[PlasmidMetadata, str]:
        """Resolve an element label to plasmid metadata and canonical element name."""
        normalized_query = self._normalize_element_name(element_name)
        normalized_to_key = {
            self._normalize_element_name(key): key
            for key in self.element_to_plasmid
        }

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
            raise ValueError(
                f"Element name {element_name!r} matched multiple plasmids: "
                + ", ".join(prefix_matches)
            )

        available = ", ".join(self.element_to_plasmid)
        raise ValueError(
            f"Unknown MPRA element {element_name!r}. Available elements: {available}"
        )

    def _primer_tail_element_name(self, element_name: str) -> str:
        """Resolve the element name used by the primer-tail table."""
        if self.primer_tails is None:
            return element_name

        return Plasmid._resolve_element_key(element_name, self.primer_tails)

    @staticmethod
    def _normalize_element_name(element_name: str) -> str:
        """Normalize an element name for dictionary matching."""
        return re.sub(r"[^a-z0-9]+", "", element_name.lower())


@dataclass(frozen=True)
class SafeHarborSite:
    abbrev_name: str
    chromosome: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive
    category: str = ""
    genes: str = ""


@dataclass(frozen=True)
class GenomeInterval:
    chromosome: str
    start: int  # 1-based inclusive
    end: int    # 1-based inclusive

    @property
    def insertion_index(self) -> int:
        """Return midpoint as a 0-based insertion index."""
        return ((self.start - 1) + self.end) // 2


class GenomeRegion:
    """
    Build fixed-length genomic contexts around human genome safe-harbor sites.

    The genome FASTA should be indexed for pysam, for example:
        samtools faidx hg38.fa

    Examples:
        genome = GenomeRegion("hg38.fa")
        pad_aavs1 = genome.genomic_context("AAVS1", target_length=2114)
        model_input = pad_aavs1("ACGT")

        pad_manual = genome.genomic_context(
            chromosome="chr19",
            position=55112144,
            target_length=2114,
        )

        bound = GenomeRegion("hg38.fa", "Dep.34")
        pad_dep34 = bound.genomic_context(target_length=2114)
    """

    SAFE_HARBORS: ClassVar[Dict[str, SafeHarborSite]] = {
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
    ):
        """Create a genome context builder with an optional default interval."""
        self.genome_path = Path(genome_path)
        self._default_interval: Optional[GenomeInterval] = None
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
    ) -> Callable[[str], str]:
        """
        Return a function that pads a sequence to target_length with genomic DNA.

        Coordinates are 1-based inclusive. If only `position` is supplied, the
        input sequence is inserted at that point. If `start` and `end` are
        supplied, the interval midpoint is used as the insertion point.
        """
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
        insertion_index = interval.insertion_index

        def context_for_sequence(sequence: str) -> str:
            """Pad one sequence to target length with genomic flanks."""
            sequence = self._clean_sequence(sequence)
            if len(sequence) > target_length:
                raise ValueError(
                    f"Input sequence length ({len(sequence)}) is longer than "
                    f"target_length ({target_length})"
                )

            missing = target_length - len(sequence)
            left_len = missing // 2
            right_len = missing - left_len

            left = self._fetch(interval.chromosome, insertion_index - left_len, insertion_index)
            right = self._fetch(interval.chromosome, insertion_index, insertion_index + right_len)
            return left + sequence + right

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
    ) -> Optional[GenomeInterval]:
        """Resolve named or manual coordinates into a genome interval."""
        if region is None and chromosome is None and position is None and start is None and end is None:
            default_interval = getattr(self, "_default_interval", None)
            if default_interval is not None:
                return default_interval
            if required:
                raise ValueError(
                    "Provide a safe-harbor abbrev name, e.g. 'AAVS1', or manual "
                    "coordinates with chromosome plus position/start/end."
                )
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

        available = ", ".join(cls.SAFE_HARBORS)
        raise ValueError(f"Unknown safe-harbor abbrev name {name!r}. Available names: {available}")

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
                raise ValueError(
                    f"Requested context {reference}:{start0 + 1}-{end0} exceeds "
                    f"chromosome bounds 1-{ref_len}"
                )

            return fasta.fetch(reference, start0, end0).upper()

    @staticmethod
    def _resolve_reference_name(fasta, chromosome: str) -> str:
        """Map chromosome labels to names present in the FASTA index."""
        names = set(fasta.references)
        raw = chromosome.removeprefix("chr")
        candidates = [chromosome, raw, f"chr{raw}"]
        if raw in {"M", "MT"}:
            candidates.extend(["M", "MT", "chrM", "chrMT"])

        for candidate in candidates:
            if candidate in names:
                return candidate

        raise ValueError(
            f"Chromosome {chromosome!r} was not found in FASTA. "
            f"Tried: {', '.join(dict.fromkeys(candidates))}"
        )
