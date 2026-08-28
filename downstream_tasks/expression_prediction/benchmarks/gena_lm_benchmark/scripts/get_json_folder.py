#!/usr/bin/env python

from pathlib import Path
import argparse
import os
import shutil

import pandas as pd


def resolve_metadata_path(task_root, metadata_value):
    metadata_path = Path(str(metadata_value))

    if metadata_path.is_absolute():
        return metadata_path

    # Mapping files usually store paths like ../metadata/ENCFF035CWS.json.
    if str(metadata_path).startswith("../"):
        metadata_path = Path(str(metadata_path).replace("../", "", 1))

    return task_root / metadata_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a folder with cell-line JSON files from a mapping CSV."
    )
    parser.add_argument("--map", required=True, help="CSV with id and metadata columns")
    parser.add_argument("--task-root", required=True, help="Root folder with metadata folders")
    parser.add_argument("--out-dir", required=True, help="Output JSON folder")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy JSON files instead of making symlinks",
    )
    args = parser.parse_args()

    map_path = Path(args.map)
    task_root = Path(args.task_root)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(map_path)

    if "id" not in df.columns:
        raise ValueError("Mapping CSV must contain column: id")
    if "metadata" not in df.columns:
        raise ValueError("Mapping CSV must contain column: metadata")

    created = 0
    missing = []

    for _, row in df.iterrows():
        cell_id = str(row["id"]).strip()
        src = resolve_metadata_path(task_root, row["metadata"])
        dst = out_dir / f"{cell_id}.json"

        if not src.exists():
            missing.append((cell_id, str(src)))
            continue

        if dst.exists() or dst.is_symlink():
            dst.unlink()

        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)

        created += 1

    print("mapping:", map_path)
    print("output:", out_dir)
    print("created:", created)
    print("missing:", len(missing))

    if missing:
        print("first missing files:")
        for cell_id, path in missing[:20]:
            print(cell_id, path)


if __name__ == "__main__":
    main()
