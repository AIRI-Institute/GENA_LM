#!/usr/bin/env python3
"""Dump AlphaGenome human RNA-seq track metadata to CSV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from alphagenome.models import dna_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--api-key-env",
        default="ALPHAGENOME_API_KEY",
        help="Environment variable containing the AlphaGenome API key.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running this script.")

    model = dna_client.create(api_key)
    metadata = model.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
    rna_seq = metadata.rna_seq
    if rna_seq is None:
        raise SystemExit("AlphaGenome returned no RNA-seq metadata.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rna_seq.to_csv(out, index=False)
    print(f"Wrote {out} with {len(rna_seq)} RNA-seq tracks.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
