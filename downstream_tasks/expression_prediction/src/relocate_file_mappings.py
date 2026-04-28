#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PATH_COLUMN_HINTS = (
    "bam",
    "bed",
    "bedgraph",
    "bw",
    "csv",
    "embedding",
    "forward",
    "genome",
    "metadata",
    "path",
    "pkl",
    "reverse",
    "text_data",
    "tpm",
    "tsv",
)

KNOWN_PATH_SUFFIXES = (
    ".bam",
    ".bed",
    ".bedgraph",
    ".bigwig",
    ".bw",
    ".csv",
    ".fa",
    ".fasta",
    ".gz",
    ".h5",
    ".json",
    ".pkl",
    ".tsv",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in config: {path}")
    return data


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def resolve_home_path(raw_path: str, repo: Path) -> Path:
    raw_path = str(raw_path)
    home_dir = repo.parent
    gena_prefix = "${HOME_PATH}/GENA_LM/"
    home_prefix = "${HOME_PATH}/"

    if raw_path.startswith(gena_prefix):
        return repo / raw_path[len(gena_prefix) :]
    if raw_path.startswith(home_prefix):
        return home_dir / raw_path[len(home_prefix) :]
    if raw_path.startswith("~/"):
        return Path(raw_path).expanduser()
    return Path(raw_path)


def build_file_mappings_dir_raw(intervals_path_raw: str) -> str:
    intervals_path = PurePosixPath(intervals_path_raw)
    return str(intervals_path.parent.parent / "file_mappings")


def looks_like_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("http://", "https://", "s3://")):
        return False
    if (
        value.startswith(("/", ".", "~", "${"))
        or "/" in value
        or "\\" in value
    ):
        return True
    return value.lower().endswith(KNOWN_PATH_SUFFIXES)


def key_suggests_path(key: str) -> bool:
    key_lower = key.lower()
    return any(hint in key_lower for hint in PATH_COLUMN_HINTS)


def should_rewrite_path(key: str, value: str) -> bool:
    if not value:
        return False
    if looks_like_path(value):
        return True
    return key_suggests_path(key) and value.lower().endswith(KNOWN_PATH_SUFFIXES)


def rewrite_path_value(value: str, src: Path, dst: Path, repo: Path) -> str:
    resolved = resolve_home_path(value, repo)
    if not resolved.is_absolute():
        resolved = src.parent / resolved
    rel = os.path.relpath(resolved.resolve(), start=dst.parent)
    return rel


def rewrite_mapping_csv(src: Path, dst: Path, repo: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(
            f"Destination file_mappings already exists: {dst}. "
            "Move or remove it before running again."
        )

    with src.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No CSV header found in {src}")
        rows = list(reader)

    rewritten_rows: list[dict[str, str]] = []
    for row in rows:
        new_row: dict[str, str] = {}
        for key, value in row.items():
            if value is None or value == "":
                new_row[key] = value
                continue

            if should_rewrite_path(key, value):
                new_row[key] = rewrite_path_value(value, src, dst, repo)
            else:
                new_row[key] = value
        rewritten_rows.append(new_row)

    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rewritten_rows)


def iter_dataset_blocks(cfg: dict[str, Any]):
    for name, value in cfg.items():
        if not isinstance(value, dict):
            continue
        if "targets_path" not in value:
            continue
        if "forward_intervals_path" not in value and "reverse_intervals_path" not in value:
            continue
        yield name, value


def transform_config(config_path: Path) -> tuple[Path, Path]:
    repo = repo_root()
    cfg = load_yaml(config_path)

    backup_path = config_path.with_name(f"{config_path.stem}_old{config_path.suffix}")
    if backup_path.exists():
        raise FileExistsError(
            f"Backup config already exists: {backup_path}. "
            "Move or remove it before running again."
        )

    for dataset_name, block in iter_dataset_blocks(cfg):
        targets_path_raw = block["targets_path"]
        intervals_path_raw = block.get("forward_intervals_path") or block.get("reverse_intervals_path")

        src_mapping = resolve_home_path(targets_path_raw, repo).resolve()
        intervals_path = resolve_home_path(intervals_path_raw, repo).resolve()
        file_mappings_dir = intervals_path.parent.parent / "file_mappings"

        dst_mapping_name = f"file_mappings_{dataset_name}.csv"
        dst_mapping = file_mappings_dir / dst_mapping_name
        rewrite_mapping_csv(src_mapping, dst_mapping, repo)

        new_targets_dir_raw = build_file_mappings_dir_raw(intervals_path_raw)
        block["targets_path"] = str(PurePosixPath(new_targets_dir_raw) / dst_mapping_name)

    config_path.rename(backup_path)
    dump_yaml(cfg, config_path)
    return backup_path, config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy all targets_path file_mappings referenced by a final config into a "
            "sibling file_mappings directory next to intervals, rewrite relative paths "
            "inside those CSVs, rename final.yaml -> final_old.yaml, and write a new final.yaml."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "final.yaml",
        help="Path to the final.yaml config to rewrite",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backup_path, new_config_path = transform_config(args.config.resolve())
    print(f"Moved original config to: {backup_path}")
    print(f"Wrote updated config to: {new_config_path}")


if __name__ == "__main__":
    main()
