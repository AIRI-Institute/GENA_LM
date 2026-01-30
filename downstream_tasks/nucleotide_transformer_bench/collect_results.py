#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import pandas as pd
import numpy as np


def parse_sci_token(s: str) -> float | None:
    s = s.strip()
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)e(-?[0-9]+)", s, flags=re.IGNORECASE)
    if not m:
        return None
    mant = float(m.group(1))
    exp_str = m.group(2)
    exp = int(exp_str)
    if "-" not in exp_str and exp > 0:
        exp = -exp
    return mant * (10 ** exp)


def parse_run_params(run_name: str) -> dict:
    params = {"lr": np.nan, "wd": np.nan, "bs": np.nan}
    for t in run_name.split("_"):
        if t.startswith("lr"):
            v = parse_sci_token(t[2:])
            if v is not None:
                params["lr"] = v
        elif t.startswith("wd"):
            v = parse_sci_token(t[2:])
            if v is not None:
                params["wd"] = v
        elif t.startswith("bs"):
            m = re.search(r"\d+", t)
            if m:
                params["bs"] = int(m.group(0))
    return params


def fold_from_dirname(name: str) -> int | None:
    m = re.fullmatch(r"fold_(\d+)", name)
    return int(m.group(1)) if m else None


def fmt_float_comma(x, nd=3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}".replace(".", ",")


def fmt_sci_comma(x) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.2E}".replace(".", ",")


def build_raw_table(root_dir: Path, n_folds: int) -> pd.DataFrame:
    rows = []
    for results_dir in root_dir.rglob("results"):
        if not results_dir.is_dir():
            continue

        json_files = sorted(results_dir.glob("*.json"))
        if len(json_files) != 1:
            continue

        json_path = json_files[0]
        fold_dir = results_dir.parent
        run_dir = fold_dir.parent
        task_dir = run_dir.parent

        fold = fold_from_dirname(fold_dir.name)
        task = task_dir.name
        params = parse_run_params(run_dir.name)

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        mcc = None
        if isinstance(data, dict):
            mm = data.get("metrics", {})
            if isinstance(mm, dict):
                mcc = mm.get("mcc", None)

        rows.append(
            {
                "task": task,
                "lr": params["lr"],
                "wd": params["wd"],
                "bs": params["bs"],
                "fold": fold,
                "mcc": mcc,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["mcc"] = pd.to_numeric(df["mcc"], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce").astype("Int64")

    wide = df.pivot_table(
        index=["task", "lr", "wd", "bs"],
        columns="fold",
        values="mcc",
        aggfunc="first",
    )

    fold_cols = list(range(n_folds))
    wide = wide.reindex(columns=fold_cols)

    wide["mean"] = wide[fold_cols].mean(axis=1, skipna=True)
    wide["std"] = wide[fold_cols].std(axis=1, ddof=1, skipna=True)

    out = wide.reset_index()
    col_order = ["task", "mean", "std", "lr", "wd", "bs"] + fold_cols
    out = out[col_order].sort_values(by=["task", "lr", "wd", "bs"], kind="mergesort").reset_index(drop=True)
    return out


def format_table(out_raw: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    if out_raw.empty:
        return out_raw

    fold_cols = list(range(n_folds))
    out_fmt = out_raw.copy()

    out_fmt["mean"] = out_fmt["mean"].apply(lambda x: fmt_float_comma(x, 3))
    out_fmt["std"] = out_fmt["std"].apply(lambda x: fmt_float_comma(x, 3))
    for c in fold_cols:
        if c in out_fmt.columns:
            out_fmt[c] = out_fmt[c].apply(lambda x: fmt_float_comma(x, 3))

    out_fmt["lr"] = out_fmt["lr"].apply(fmt_sci_comma)
    out_fmt["wd"] = out_fmt["wd"].apply(fmt_sci_comma)
    out_fmt["bs"] = out_fmt["bs"].apply(lambda x: "" if pd.isna(x) else str(int(x)))

    out_fmt.columns.name = None
    return out_fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--out", type=str, default="results.csv")
    args = ap.parse_args()

    root_dir = Path(args.root)
    out_path = Path(args.out)

    out_raw = build_raw_table(root_dir, n_folds=args.folds)
    out_fmt = format_table(out_raw, n_folds=args.folds)

    out_fmt = out_fmt.sort_values(by=["task"], kind="mergesort").reset_index(drop=True)
    out_fmt.to_csv(out_path, index=False)

    print(str(out_path.resolve()))


if __name__ == "__main__":
    main()
