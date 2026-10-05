#!/usr/bin/env python3
"""Build REF/ALT 9 kbp sequence pairs for Liver.train.v2.csv.

Each output record corresponds to one credible-set variant and one gene from
``genes_with_zero_expression``. The record contains a 9 kbp REF sequence and a
matched ALT sequence:

    3 kbp variant-centered region + 6 kbp gene-TSS-centered region

Coordinates are 1-based B37 in the input files. Pass a B37/hg19 FASTA.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VARIANT_REGION_LEN = 3_000
GENE_REGION_LEN = 6_000


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int
    ref: str
    alt: str

    @property
    def variant_id(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.ref}-{self.alt}"


class FastaGenome:
    """Small pyfaidx wrapper using 0-based half-open fetch coordinates."""

    def __init__(self, fasta_path: str | Path):
        try:
            from pyfaidx import Fasta
        except ImportError:
            self.fasta = FaiIndexedFasta(fasta_path)
            self._chrom_cache: dict[str, str] = {}
            return

        self.fasta = Fasta(str(fasta_path), sequence_always_upper=True)
        self._chrom_cache: dict[str, str] = {}

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        """Return uppercase sequence for [start0, end0), padded with N."""
        if end0 <= start0:
            return ""

        left_pad = max(0, -start0)
        fetch_start = max(0, start0)
        fetch_end = max(fetch_start, end0)
        fasta_chrom = self._resolve_chrom(chrom)
        seq = str(self.fasta[fasta_chrom][fetch_start:fetch_end]).upper()
        seq = "".join(base if base in "ACGT" else "N" for base in seq)

        target_len = end0 - start0
        padded = ("N" * left_pad) + seq
        if len(padded) < target_len:
            padded += "N" * (target_len - len(padded))
        return padded[:target_len]

    def _resolve_chrom(self, chrom: str) -> str:
        if chrom in self._chrom_cache:
            return self._chrom_cache[chrom]
        candidates = [chrom]
        if chrom.startswith("chr"):
            candidates.append(chrom[3:])
        else:
            candidates.append(f"chr{chrom}")
        for candidate in candidates:
            if candidate in self.fasta:
                self._chrom_cache[chrom] = candidate
                return candidate
        raise KeyError(f"Chromosome {chrom!r} is not present in FASTA")


@dataclass(frozen=True)
class FaiRecord:
    length: int
    offset: int
    line_bases: int
    line_width: int


class FaiIndexedFasta:
    """Minimal uncompressed FASTA reader backed by a samtools .fai index."""

    def __init__(self, fasta_path: str | Path):
        self.fasta_path = Path(fasta_path)
        self.fai_path = Path(f"{self.fasta_path}.fai")
        if not self.fai_path.exists():
            raise FileNotFoundError(
                f"pyfaidx is not installed and FASTA index is missing: {self.fai_path}"
            )
        self.records = self._load_fai(self.fai_path)
        self.handle = self.fasta_path.open("rb")

    def __contains__(self, chrom: str) -> bool:
        return chrom in self.records

    def __getitem__(self, chrom: str):
        return FaiContig(self, chrom)

    @staticmethod
    def _load_fai(fai_path: Path) -> dict[str, FaiRecord]:
        records: dict[str, FaiRecord] = {}
        with fai_path.open() as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    raise ValueError(f"Malformed .fai line: {line!r}")
                records[fields[0]] = FaiRecord(
                    length=int(fields[1]),
                    offset=int(fields[2]),
                    line_bases=int(fields[3]),
                    line_width=int(fields[4]),
                )
        return records

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        record = self.records[chrom]
        clipped_start = max(0, min(start0, record.length))
        clipped_end = max(clipped_start, min(end0, record.length))
        if clipped_end == clipped_start:
            return ""

        byte_start = self._byte_offset(record, clipped_start)
        byte_end = self._byte_offset(record, clipped_end)
        self.handle.seek(byte_start)
        raw = self.handle.read(byte_end - byte_start)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"").decode("ascii").upper()
        return seq[:clipped_end - clipped_start]

    @staticmethod
    def _byte_offset(record: FaiRecord, pos0: int) -> int:
        return (
            record.offset
            + (pos0 // record.line_bases) * record.line_width
            + (pos0 % record.line_bases)
        )


class FaiContig:
    def __init__(self, fasta: FaiIndexedFasta, chrom: str):
        self.fasta = fasta
        self.chrom = chrom

    def __getitem__(self, key: slice) -> str:
        if not isinstance(key, slice):
            raise TypeError("FASTA contig access requires a slice")
        start = 0 if key.start is None else key.start
        end = self.fasta.records[self.chrom].length if key.stop is None else key.stop
        return self.fasta.fetch(self.chrom, start, end)


def split_csv_cell(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_chrom(value: str) -> str:
    value = value.strip()
    return value if value.startswith("chr") else f"chr{value}"


def parse_variant_token(token: str, default_chrom: str | None = None) -> Variant:
    parts = token.strip().split("-")
    if len(parts) != 4:
        raise ValueError(f"Expected CHR-POS-REF-ALT variant token, got {token!r}")

    chrom = normalize_chrom(parts[0] or default_chrom or "")
    if chrom == "chr":
        raise ValueError(f"Missing chromosome in variant token {token!r}")

    ref = parts[2].upper()
    alt = parts[3].upper()
    if not ref or not alt:
        raise ValueError(f"Missing REF or ALT allele in variant token {token!r}")

    return Variant(chrom=chrom, pos=int(parts[1]), ref=ref, alt=alt)


def centered_start0(center_pos1: int, length: int) -> int:
    """0-based start for an even-length window with center at length // 2."""
    if center_pos1 < 1:
        raise ValueError(f"1-based center position must be positive, got {center_pos1}")
    if length < 1:
        raise ValueError(f"Window length must be positive, got {length}")
    return center_pos1 - (length // 2) - 1


def fetch_centered_sequence(
    genome,
    chrom: str,
    center_pos1: int,
    length: int,
    extra_after: int = 0,
) -> tuple[str, int]:
    start0 = centered_start0(center_pos1, length)
    seq = genome.fetch(chrom, start0, start0 + length + extra_after)
    return seq, start0


def build_alt_variant_region(
    variant_ref_region: str,
    variant_offset: int,
    ref_allele: str,
    alt_allele: str,
    target_len: int,
    strict_ref_match: bool = True,
) -> str:
    observed_ref = variant_ref_region[variant_offset:variant_offset + len(ref_allele)]
    if strict_ref_match and observed_ref != ref_allele:
        raise ValueError(
            "FASTA REF allele mismatch: "
            f"expected {ref_allele!r}, observed {observed_ref!r} at offset {variant_offset}"
        )

    alt_region = (
        variant_ref_region[:variant_offset]
        + alt_allele
        + variant_ref_region[variant_offset + len(ref_allele):]
    )
    if len(alt_region) < target_len:
        alt_region += "N" * (target_len - len(alt_region))
    return alt_region[:target_len]


def load_gene_tss(path: str | Path) -> dict[str, int]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        required = {"gene_id", "TSS_B37"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return {
            row["gene_id"]: int(row["TSS_B37"])
            for row in reader
            if row.get("gene_id") and row.get("TSS_B37")
        }


def build_liver_llm_sequence_records(
    liver_train_csv: str | Path,
    gene_tss_csv: str | Path,
    genome,
    variant_region_len: int = VARIANT_REGION_LEN,
    gene_region_len: int = GENE_REGION_LEN,
    max_rows: int | None = None,
    strict_ref_match: bool = True,
    skip_missing_tss: bool = False,
) -> list[dict[str, object]]:
    return list(
        iter_liver_llm_sequence_records(
            liver_train_csv=liver_train_csv,
            gene_tss_csv=gene_tss_csv,
            genome=genome,
            variant_region_len=variant_region_len,
            gene_region_len=gene_region_len,
            max_rows=max_rows,
            strict_ref_match=strict_ref_match,
            skip_missing_tss=skip_missing_tss,
            missing_tss_genes=set(),
        )
    )


def iter_liver_llm_sequence_records(
    liver_train_csv: str | Path,
    gene_tss_csv: str | Path,
    genome,
    variant_region_len: int = VARIANT_REGION_LEN,
    gene_region_len: int = GENE_REGION_LEN,
    max_rows: int | None = None,
    strict_ref_match: bool = True,
    skip_missing_tss: bool = False,
    missing_tss_genes: set[str] | None = None,
):
    """Yield paired REF/ALT sequence records without keeping all rows in memory."""
    gene_tss = load_gene_tss(gene_tss_csv)
    if missing_tss_genes is None:
        missing_tss_genes = set()

    with Path(liver_train_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{liver_train_csv} has no CSV header")

        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max_rows:
                break

            signal_id = row["common_variant_analysis_signal_id"]
            row_chrom = normalize_chrom(row["chr"])
            variants = [
                parse_variant_token(token, default_chrom=row_chrom)
                for token in split_csv_cell(row["credible_set_variants_in_1_based_B37"])
            ]
            genes = split_csv_cell(row["genes_with_zero_expression"])

            for variant in variants:
                if variant.chrom != row_chrom:
                    raise ValueError(
                        f"Row {signal_id} has chr={row_chrom}, but variant is {variant.variant_id}"
                    )

                extra_after = max(0, len(variant.ref) - len(variant.alt))
                variant_ref_extra, variant_start0 = fetch_centered_sequence(
                    genome,
                    variant.chrom,
                    variant.pos,
                    variant_region_len,
                    extra_after=extra_after,
                )
                variant_ref = variant_ref_extra[:variant_region_len]
                variant_offset = (variant.pos - 1) - variant_start0
                variant_alt = build_alt_variant_region(
                    variant_ref_extra,
                    variant_offset,
                    variant.ref,
                    variant.alt,
                    variant_region_len,
                    strict_ref_match=strict_ref_match,
                )

                for gene_id in genes:
                    if gene_id not in gene_tss:
                        missing_tss_genes.add(gene_id)
                        if skip_missing_tss:
                            continue
                        raise KeyError(f"Gene {gene_id!r} is absent from {gene_tss_csv}")

                    tss = gene_tss[gene_id]
                    gene_region, gene_start0 = fetch_centered_sequence(
                        genome,
                        variant.chrom,
                        tss,
                        gene_region_len,
                    )
                    ref_sequence = variant_ref + gene_region
                    alt_sequence = variant_alt + gene_region

                    yield {
                        "common_variant_analysis_signal_id": signal_id,
                        "row_index": row_index,
                        "chrom": variant.chrom,
                        "variant_pos_b37": variant.pos,
                        "variant_id": variant.variant_id,
                        "ref_allele": variant.ref,
                        "alt_allele": variant.alt,
                        "gene_id": gene_id,
                        "gene_tss_b37": tss,
                        "true_gene": row.get("true_gene", ""),
                        "eqtl_tissue": row.get("eqtl_tissue", ""),
                        "category": row.get("category", ""),
                        "variant_region_start_0based": variant_start0,
                        "gene_region_start_0based": gene_start0,
                        "variant_offset_in_sequence": variant_offset,
                        "gene_tss_offset_in_sequence": (
                            variant_region_len + ((tss - 1) - gene_start0)
                        ),
                        "ref_sequence": ref_sequence,
                        "alt_sequence": alt_sequence,
                    }

    if missing_tss_genes and skip_missing_tss:
        preview = ", ".join(sorted(missing_tss_genes)[:10])
        print(
            f"Skipped {len(missing_tss_genes)} genes absent from {gene_tss_csv}: {preview}",
            flush=True,
        )


def build_liver_llm_sequence_dataframe(*args, **kwargs):
    """Return the paired REF/ALT output as a pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for DataFrame output") from exc
    return pd.DataFrame(build_liver_llm_sequence_records(*args, **kwargs))


def to_long_records(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    long_records: list[dict[str, object]] = []
    for record in records:
        base = {k: v for k, v in record.items() if k not in {"ref_sequence", "alt_sequence"}}
        for allele_type, sequence_key in (("ref", "ref_sequence"), ("alt", "alt_sequence")):
            long_record = dict(base)
            long_record["allele_type"] = allele_type
            long_record["sequence"] = record[sequence_key]
            long_record["sequence_id"] = (
                f"{record['common_variant_analysis_signal_id']}|"
                f"{record['variant_id']}|{record['gene_id']}|{allele_type}"
            )
            long_records.append(long_record)
    return long_records


def write_records(path: str | Path, records: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        return

    suffixes = [suffix.lower() for suffix in out_path.suffixes]
    delimiter = "\t" if ".tsv" in suffixes or ".txt" in suffixes else ","
    opener = gzip.open if out_path.suffix.lower() == ".gz" else open
    with opener(out_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0]) if records else [],
            delimiter=delimiter,
        )
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--liver-train-csv",
        default="data/Attachments_robert_2/Liver.train.v2.csv",
    )
    parser.add_argument(
        "--gene-tss-csv",
        default="data/Attachments_robert_2/gene_tss.v2.csv",
    )
    parser.add_argument("--b37-fasta", required=True, help="B37/hg19 reference FASTA")
    parser.add_argument("--out", required=True, help="Output .csv, .tsv, .csv.gz, .tsv.gz, or .jsonl")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--long",
        action="store_true",
        help="Write one row per allele sequence instead of paired REF/ALT rows",
    )
    parser.add_argument(
        "--allow-ref-mismatch",
        action="store_true",
        help="Continue when FASTA bases do not match REF alleles in the CSV",
    )
    parser.add_argument(
        "--skip-missing-tss",
        action="store_true",
        help="Skip genes absent from gene_tss.v2.csv instead of failing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genome = FastaGenome(args.b37_fasta)
    records = build_liver_llm_sequence_records(
        liver_train_csv=args.liver_train_csv,
        gene_tss_csv=args.gene_tss_csv,
        genome=genome,
        max_rows=args.max_rows,
        strict_ref_match=not args.allow_ref_mismatch,
        skip_missing_tss=args.skip_missing_tss,
    )
    output_records = to_long_records(records) if args.long else records
    write_records(args.out, output_records)
    print(
        f"Wrote {len(output_records)} rows to {args.out} "
        f"({2 * len(records)} biological sequences)"
    )


if __name__ == "__main__":
    main()
