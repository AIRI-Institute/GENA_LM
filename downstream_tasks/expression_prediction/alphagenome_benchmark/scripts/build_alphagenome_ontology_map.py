#!/usr/bin/env python3
"""Resolve ENCODE RNA-seq accessions to ontology CURIEs for AlphaGenome.

The qnorm map contains ENCODE accessions, while AlphaGenome filters output
tracks by biological ontology CURIEs such as UBERON:0001159 or CL:0000084.
This script queries ENCODE metadata and writes a table that bridges those IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


ENCODE_BASE = "https://www.encodeproject.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qnorm-map", required=True, help="CSV with id/original_id/targets_identifier_base")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--cache", required=True, help="JSON cache for ENCODE API responses")
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between uncached ENCODE requests")
    parser.add_argument("--retries", type=int, default=3, help="Retries for transient ENCODE failures")
    return parser.parse_args()


def load_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open() as handle:
            return json.load(handle)
    return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def encode_get(accession: str, cache: dict[str, Any], retries: int, sleep: float) -> dict[str, Any]:
    if accession in cache:
        return cache[accession]

    if accession.startswith("ENCFF"):
        url = f"{ENCODE_BASE}/files/{accession}/?format=json"
    elif accession.startswith("ENCSR"):
        url = f"{ENCODE_BASE}/experiments/{accession}/?format=json"
    else:
        raise ValueError(f"Unsupported ENCODE accession: {accession}")

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                headers={"accept": "application/json"},
                timeout=(10, 60),
            )
            response.raise_for_status()
            cache[accession] = response.json()
            time.sleep(sleep)
            return cache[accession]
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(sleep * attempt * 5)
    raise RuntimeError(f"Failed to fetch {accession}: {last_error}") from last_error


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def extract_ontology(payload: dict[str, Any]) -> tuple[str, str, str, str]:
    ontology = payload.get("biosample_ontology") or {}
    term_id = ontology.get("term_id") or ""
    term_name = ontology.get("term_name") or ""
    classification = ontology.get("classification") or ""
    assay = payload.get("assay_title") or payload.get("assay_term_name") or ""
    return term_id, term_name, classification, assay


def candidate_accessions(row: dict[str, str]) -> list[str]:
    candidates = [
        row.get("id", ""),
        row.get("targets_identifier_base", ""),
        row.get("original_id", ""),
    ]
    seen: set[str] = set()
    unique = []
    for accession in candidates:
        accession = accession.strip()
        if accession and accession not in seen and accession.startswith(("ENCFF", "ENCSR")):
            unique.append(accession)
            seen.add(accession)
    return unique


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)

    with Path(args.qnorm_map).open() as handle:
        rows = list(csv.DictReader(handle))

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        resolved: dict[str, str] | None = None
        errors: list[str] = []

        for accession in candidate_accessions(row):
            try:
                payload = encode_get(accession, cache, args.retries, args.sleep)
                term_id, term_name, classification, assay = extract_ontology(payload)
                if term_id:
                    resolved = {
                        "query_accession": accession,
                        "query_type": "file" if accession.startswith("ENCFF") else "experiment",
                        "ontology_curie": term_id,
                        "biosample_name": term_name,
                        "biosample_classification": classification,
                        "assay": assay,
                    }
                    break
                errors.append(f"{accession}: no biosample_ontology.term_id")
            except Exception as exc:  # Keep going; one alternate accession may work.
                errors.append(f"{accession}: {exc}")

        if resolved is None:
            resolved = {
                "query_accession": "",
                "query_type": "",
                "ontology_curie": "",
                "biosample_name": "",
                "biosample_classification": "",
                "assay": "",
            }

        output_rows.append(
            {
                "row_index": str(index),
                "id": row.get("id", ""),
                "original_id": row.get("original_id", ""),
                "targets_identifier_base": row.get("targets_identifier_base", ""),
                **resolved,
                "errors": " | ".join(errors),
            }
        )

        if index % 25 == 0:
            save_cache(cache_path, cache)
            print(f"resolved {index}/{len(rows)}", file=sys.stderr)

    save_cache(cache_path, cache)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    resolved_count = sum(bool(row["ontology_curie"]) for row in output_rows)
    print(f"Wrote {out_path} with {resolved_count}/{len(output_rows)} resolved rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
