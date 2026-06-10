from __future__ import annotations

import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, ClassVar


InsertFn = Callable[[str, str], str]


table18_primers: Dict[str, Tuple[str, str]] = {
    "F9": ("CCCGGGCTCGAGATCTCCCACTGATGAACTGTGC", "CCGGATTGCCAAGCTTAACCTTTGCTAGCAGATTGTG"),
    "FOXE1": ("CCCGGGCTCGAGATCTCTCGCCAGCGGTCCGCAGG", "CCGGATTGCCAAGCTTGGCCTGGCGTCCCCGGAACG"),
    "GP1BB": ("CCCGGGCTCGAGATCTTTGTGAATGCCGCGTCCTG", "CCGGATTGCCAAGCTTACGACCAGAGCTCCTCTC"),
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
    "GP1BB": (19723266, 19723650),
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
    "GP1BB": "22",
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
        reporter_feature_name: str,
    ):
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
        parsed = urllib.parse.urlparse(ncbi_url)
        accession = parsed.path.rstrip("/").split("/")[-1]
        if not accession:
            raise ValueError(f"Could not parse accession from URL: {ncbi_url}")
        return accession

    @staticmethod
    def _download_text(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "plasmid-context/1.0"})
        with urllib.request.urlopen(req) as response:
            return response.read().decode("utf-8")

    def _ncbi_sviewer_url(self, report: str) -> str:
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
        key = re.sub(r"\s+", " ", text.strip())
        return key if key else "unnamed_feature"

    @staticmethod
    def _read_fasta_sequence(path: Path) -> str:
        lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(">"):
                lines.append(line.strip())
        return "".join(lines).upper()

    @staticmethod
    def _parse_locus_name(genbank_text: str) -> Optional[str]:
        match = re.search(r"^LOCUS\s+(\S+)", genbank_text, flags=re.MULTILINE)
        return match.group(1) if match else None

    @staticmethod
    def _parse_location(location: str) -> Tuple[int, int, int]:
        strand = -1 if location.startswith("complement(") else 1
        numbers = [int(number) for number in re.findall(r"\d+", location)]
        if not numbers:
            raise ValueError(f"Could not parse feature location: {location}")
        return min(numbers), max(numbers), strand

    @staticmethod
    def _parse_qualifiers(block: str) -> Dict[str, str]:
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
        features: Dict[str, Tuple[int, int]] = {}
        counts: Dict[str, int] = {}

        for feature in self._feature_records:
            label = feature["label"]
            counts[label] = counts.get(label, 0) + 1
            key = label if counts[label] == 1 else f"{label}#{counts[label]}"
            features[key] = (feature["start"], feature["end"])

        return features

    def _find_reporter_feature(self, reporter_feature_name: str) -> FeatureMatch:
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

        if len(matches) > 1:
            exact = [m for m in matches if m.key.lower() == needle]
            if len(exact) == 1:
                return exact[0]
            raise ValueError(
                f"Reporter feature name {reporter_feature_name!r} matched multiple features: "
                + ", ".join(m.key for m in matches)
            )

        return matches[0]

    def reporter_tss_index(self) -> int:
        """Return reporter TSS as 0-based sequence index."""
        feature = self.reporter_feature
        return feature.start - 1 if feature.strand >= 0 else feature.end - 1

    def plasmid_context(
        self,
        insert_fn: InsertFn,
        context_size: int,
        circular: bool = True,
        allow_repeats: bool = False,
    ) -> Callable[[str], str]:

        if context_size <= 0:
            raise ValueError("context_size must be positive")

        old_tss = self.reporter_tss_index()
        old_reporter_anchor = self.sequence[old_tss : old_tss + 40]

        def context_for_mpra(mpra_fragment: str) -> str:
            edited = insert_fn(self.sequence, mpra_fragment.upper()).upper()
            new_tss = edited.find(old_reporter_anchor)

            if new_tss == -1:
                new_tss = old_tss + (len(edited) - len(self.sequence))

            return self._centered_window(
                sequence=edited,
                center=new_tss,
                size=context_size,
                circular=circular,
                fill_sequence=None if not allow_repeats else self.sequence,
                fill_center=old_tss,
            )

        return context_for_mpra

    @staticmethod
    def _centered_window(
        sequence: str,
        center: int,
        size: int,
        circular: bool,
        fill_sequence: str | None = None,
        fill_center: int | None = None,
    ) -> str:
        half_left = size // 2

        if circular:
            # Old behavior: repeat the whole edited plasmid, including insert.
            if fill_sequence is None:
                n = len(sequence)
                
                new_center = center
                
                target = n // 2
                while new_center != target:
                    sequence  = sequence[n-1] + sequence[0:-1]
                    new_center += 1
                
                return sequence
            
                #start = center - half_left
                #return "".join(sequence[i % n] for i in range(start, start + size))

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
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(table)[::-1].upper()


def _best_primer_suffix_tail(
    primer: str,
    target: str,
    expected_position: int,
    min_match: int = 12,
    edge_slop: int = 25,
) -> Tuple[str, str, int]:
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
        "GP1BB": ("CCCGGGCTCGAGATCTTTGTGAATGCCGCGTCCTG", "CCGGATTGCCAAGCTTACGACCAGAGCTCCTCTC"),
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
        "GP1BB": ("22", 19723266, 19723650),
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
        plasmid_sequence = plasmid_sequence.upper()
        mpra_fragment = mpra_fragment.upper()
        assembled_insert = left_tail + mpra_fragment + right_scar

        left = plasmid_sequence.find(left_tail)
        right = plasmid_sequence.find(right_scar, left + len(left_tail))

        if left == -1 or right == -1:
            raise ValueError(
                "Could not find MCS scars in plasmid. Check tails, orientation, "
                "or whether this exact plasmid already contains the altered MCS."
            )

        return plasmid_sequence[:left] + assembled_insert + plasmid_sequence[right + len(right_scar):]

    return insert_into_mcs


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
        normalized = cls._normalize_name(name)
        for key, site in cls.SAFE_HARBORS.items():
            if normalized in {cls._normalize_name(key), cls._normalize_name(site.abbrev_name)}:
                return site

        available = ", ".join(cls.SAFE_HARBORS)
        raise ValueError(f"Unknown safe-harbor abbrev name {name!r}. Available names: {available}")

    @staticmethod
    def _normalize_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    @staticmethod
    def _clean_sequence(sequence: str) -> str:
        sequence = re.sub(r"\s+", "", sequence).upper()
        if not sequence:
            raise ValueError("Input sequence is empty")
        invalid = sorted(set(sequence) - set("ACGTN"))
        if invalid:
            raise ValueError(f"Input sequence contains non-DNA characters: {''.join(invalid)}")
        return sequence

    def _fetch(self, chromosome: str, start0: int, end0: int) -> str:
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
