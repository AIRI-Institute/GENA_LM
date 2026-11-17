#!/usr/bin/env python3
"""Run ExpressionCounts on REF/ALT sequence pairs and save predictions.

Input is the paired table produced by ``scripts/liver_llm_sequences.py``:
one row per variant/gene pair with ``ref_sequence`` and ``alt_sequence``.

The model call follows ``notebooks/inference.ipynb``:
``output["logits"][:, 0, 0]`` is used as sequence-level expression.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator
import numpy as np

def read_table(path: str | Path) -> list[dict[str, str]]:
    in_path = Path(path)
    suffixes = [suffix.lower() for suffix in in_path.suffixes]
    delimiter = "\t" if ".tsv" in suffixes or ".txt" in suffixes else ","
    opener = gzip.open if in_path.suffix.lower() == ".gz" else open
    with opener(in_path, "rt", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_description(path: str | Path) -> str:
    description = Path(path).read_text(encoding="utf-8").strip()
    if not description:
        raise ValueError(f"Description file is empty: {path}")
    return description


def write_table(path: str | Path, rows: list[dict[str, object]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [suffix.lower() for suffix in out_path.suffixes]
    delimiter = "\t" if ".tsv" in suffixes or ".txt" in suffixes else ","
    opener = gzip.open if out_path.suffix.lower() == ".gz" else open
    fieldnames = list(rows[0]) if rows else []
    with opener(out_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def output_table_writer(path: str | Path, fieldnames: list[str]):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [suffix.lower() for suffix in out_path.suffixes]
    delimiter = "\t" if ".tsv" in suffixes or ".txt" in suffixes else ","
    opener = gzip.open if out_path.suffix.lower() == ".gz" else open
    handle = opener(out_path, "wt", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    return handle, writer


def validate_sequence(sequence: str, row_name: str) -> str:
    sequence = sequence.strip().upper()
    invalid = set(sequence) - set("ACGTN")
    if invalid:
        raise ValueError(f"{row_name} contains invalid DNA bases: {sorted(invalid)}")
    return sequence


def load_sequence_pairs(path: str | Path) -> list[dict[str, object]]:
    rows = read_table(path)
    required = {"ref_sequence", "alt_sequence"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns {sorted(required)}")

    pairs: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        ref_sequence = validate_sequence(row["ref_sequence"], f"row {i} ref_sequence")
        alt_sequence = validate_sequence(row["alt_sequence"], f"row {i} alt_sequence")
        if len(ref_sequence) != len(alt_sequence):
            raise ValueError(
                f"row {i} has unequal REF/ALT lengths: "
                f"{len(ref_sequence)} != {len(alt_sequence)}"
            )
        metadata = {k: v for k, v in row.items() if k not in {"ref_sequence", "alt_sequence"}}
        pairs.append(
            {
                "pair_index": i,
                "metadata": metadata,
                "ref_sequence": ref_sequence,
                "alt_sequence": alt_sequence,
            }
        )
    return pairs


def flatten_pairs_for_inference(
    pairs: Iterable[dict[str, object]],
) -> tuple[list[str], list[tuple[int, str]]]:
    sequences: list[str] = []
    keys: list[tuple[int, str]] = []
    for pair in pairs:
        pair_index = int(pair["pair_index"])
        sequences.append(str(pair["ref_sequence"]))
        keys.append((pair_index, "ref"))
        sequences.append(str(pair["alt_sequence"]))
        keys.append((pair_index, "alt"))
    return sequences, keys


def batched(iterator: Iterable[dict[str, object]], batch_size: int) -> Iterator[list[dict[str, object]]]:
    batch: list[dict[str, object]] = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def import_object(import_path: str):
    module_name, object_name = import_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def load_hydra_config(config_path: str | Path):
    from hydra import compose, initialize_config_dir

    config_path = Path(config_path).expanduser().resolve()
    with initialize_config_dir(str(config_path.parent), version_base=None):
        return compose(config_name=config_path.name)


def instantiate_model(config, checkpoint_path: str | Path, gena_lm_home: str | Path, device):
    import torch
    from hydra.utils import instantiate

    gena_lm_home = Path(gena_lm_home).expanduser().resolve()
    if str(gena_lm_home) not in sys.path:
        sys.path.insert(0, str(gena_lm_home))

    model_cls_path = config["args_params"].get(
        "model_cls",
        "downstream_tasks.expression_prediction.expression_model_final:ExpressionCounts",
    )
    model_cls = import_object(model_cls_path)
    model_kwargs = instantiate(config["model_kwargs"])
    model = model_cls(**model_kwargs)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def predict_expression(
    model,
    dna_tokenizer,
    text_tokenizer,
    sequences: list[str],
    description: str,
    dna_max_seq_len: int,
    text_max_seq_len: int,
    batch_size: int,
    device,
    use_amp: bool = True,
) -> list[float]:
    import torch

    predictions: list[float] = []
    for start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[start:start + batch_size]
        tokenized_dna = dna_tokenizer(
            batch_sequences,
            truncation=True,
            padding="max_length",
            max_length=dna_max_seq_len,
            return_tensors="pt",
        )
        tokenized_desc = text_tokenizer(
            [description] * len(batch_sequences),
            truncation=True,
            padding="max_length",
            max_length=text_max_seq_len,
            return_tensors="pt",
        )

        input_ids = tokenized_dna["input_ids"].to(device)
        attention_mask = tokenized_dna["attention_mask"].to(device)
        desc_input_ids = tokenized_desc["input_ids"].unsqueeze(1).to(device)
        desc_attention_mask = tokenized_desc["attention_mask"].unsqueeze(1).to(device)
        dataset_flag = torch.zeros(
            size=(input_ids.shape[0], 1),
            device=device,
            dtype=torch.bool,
        )

        autocast_enabled = bool(use_amp and device.type == "cuda")
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    desc_input_ids=desc_input_ids,
                    desc_attention_mask=desc_attention_mask,
                    dataset_flag=dataset_flag,
                )
        predictions.extend(output["logits"][:, 0, 0].detach().float().cpu().tolist())
    return predictions


def build_output_rows(
    pairs: list[dict[str, object]],
    prediction_by_key: dict[tuple[int, str], float],
    description_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in pairs:
        pair_index = int(pair["pair_index"])
        ref_expression = prediction_by_key[(pair_index, "ref")]
        alt_expression = prediction_by_key[(pair_index, "alt")]
        row = dict(pair["metadata"])
        row.update(
            {
                "description_name": description_name,
                "ref_expression": ref_expression,
                "alt_expression": alt_expression,
                "delta_alt_minus_ref": np.log(alt_expression+10e-6) - np.log(ref_expression+10e-6),
            }
        )
        rows.append(row)
    return rows


def sequence_record_to_pair(record: dict[str, object], pair_index: int) -> dict[str, object]:
    return {
        "pair_index": pair_index,
        "metadata": {
            k: v for k, v in record.items()
            if k not in {"ref_sequence", "alt_sequence"}
        },
        "ref_sequence": record["ref_sequence"],
        "alt_sequence": record["alt_sequence"],
    }


def output_fieldnames_from_metadata(metadata: dict[str, object]) -> list[str]:
    return list(metadata) + [
        "description_name",
        "ref_expression",
        "alt_expression",
        "delta_alt_minus_ref",
    ]


def run_inference_on_pairs(
    pairs: list[dict[str, object]],
    model,
    dna_tokenizer,
    text_tokenizer,
    description: str,
    description_name: str,
    dna_max_seq_len: int,
    text_max_seq_len: int,
    sequence_batch_size: int,
    device,
    use_amp: bool,
) -> list[dict[str, object]]:
    sequences, keys = flatten_pairs_for_inference(pairs)
    predictions = predict_expression(
        model=model,
        dna_tokenizer=dna_tokenizer,
        text_tokenizer=text_tokenizer,
        sequences=sequences,
        description=description,
        dna_max_seq_len=dna_max_seq_len,
        text_max_seq_len=text_max_seq_len,
        batch_size=sequence_batch_size,
        device=device,
        use_amp=use_amp,
    )
    return build_output_rows(
        pairs,
        prediction_by_key=dict(zip(keys, predictions)),
        description_name=description_name,
    )


def run_direct_liver_generation_inference(
    args,
    model,
    dna_tokenizer,
    text_tokenizer,
    config,
    device,
    description_text
) -> int:
    from liver_llm_sequences import FastaGenome, iter_liver_llm_sequence_records

    genome = FastaGenome(args.b37_fasta)
    missing_tss_genes: set[str] = set()
    generated_records = iter_liver_llm_sequence_records(
        liver_train_csv=args.liver_train_csv,
        gene_tss_csv=args.gene_tss_csv,
        genome=genome,
        max_rows=args.max_rows,
        strict_ref_match=not args.allow_ref_mismatch,
        skip_missing_tss=args.skip_missing_tss,
        missing_tss_genes=missing_tss_genes,
    )

    handle = None
    writer = None
    total_rows = 0
    try:
        for record_batch in batched(generated_records, args.pair_batch_size):
            pairs = [
                sequence_record_to_pair(record, total_rows + i)
                for i, record in enumerate(record_batch)
            ]
            output_rows = run_inference_on_pairs(
                pairs=pairs,
                model=model,
                dna_tokenizer=dna_tokenizer,
                text_tokenizer=text_tokenizer,
                description=description_text,
                description_name=args.description_name,
                dna_max_seq_len=int(config["args_params"]["input_seq_len"]),
                text_max_seq_len=int(config["shared_dataset_params"]["text_max_seq_len"]),
                sequence_batch_size=args.batch_size,
                device=device,
                use_amp=not args.no_amp,
            )
            if writer is None:
                handle, writer = output_table_writer(
                    args.out,
                    output_fieldnames_from_metadata(pairs[0]["metadata"]),
                )
            writer.writerows(output_rows)
            total_rows += len(output_rows)
            if total_rows % args.log_interval == 0:
                print(f"Predicted {total_rows} REF/ALT pairs", flush=True)
    finally:
        if handle is not None:
            handle.close()

    if missing_tss_genes and args.skip_missing_tss:
        preview = ", ".join(sorted(missing_tss_genes)[:10])
        print(
            f"Skipped {len(missing_tss_genes)} genes absent from {args.gene_tss_csv}: {preview}",
            flush=True,
        )
    return total_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Paired REF/ALT sequence CSV/TSV")
    input_group.add_argument(
        "--direct-liver",
        action="store_true",
        help="Generate Liver REF/ALT sequences in memory and run inference directly",
    )
    parser.add_argument("--out", required=True, help="Output CSV/TSV with expression predictions")
    parser.add_argument("--config", default="notebooks/inference.yaml")
    parser.add_argument("--gena-lm-home", required=True, help="Path to the GENA_LM repo root")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint .bin path")
    parser.add_argument("--description-name", default="HepG2")
    parser.add_argument(
        "--description-path",
        required=True,
        help="Path to a text file with the experiment/cell-type description",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=64,
        help="Number of REF/ALT pairs generated before each inference/write step in direct mode",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--liver-train-csv", default="data/Attachments_robert_2/Liver.train.v2.csv")
    parser.add_argument("--gene-tss-csv", default="data/Attachments_robert_2/gene_tss.v2.csv")
    parser.add_argument("--b37-fasta", help="B37/hg19 FASTA for --direct-liver")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--allow-ref-mismatch", action="store_true")
    parser.add_argument("--skip-missing-tss", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoTokenizer

    gena_lm_home = Path(args.gena_lm_home).expanduser().resolve()
    os.environ["GENALM_HOME"] = str(gena_lm_home)
    if str(gena_lm_home) not in sys.path:
        sys.path.insert(0, str(gena_lm_home))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = load_hydra_config(args.config)
    model = instantiate_model(config, args.checkpoint, gena_lm_home, device)

    dna_tokenizer = AutoTokenizer.from_pretrained(config["args_params"]["gen_tokenizer"])
    text_tokenizer = AutoTokenizer.from_pretrained(
        config["shared_dataset_params"]["text_tokenizer"],
        padding_side="left",
    )
    description_text = read_description(args.description_path)

    if args.direct_liver:
        if not args.b37_fasta:
            raise ValueError("--b37-fasta is required with --direct-liver")
        total_rows = run_direct_liver_generation_inference(
            args=args,
            model=model,
            dna_tokenizer=dna_tokenizer,
            text_tokenizer=text_tokenizer,
            config=config,
            device=device,
            description_text = description_text
        )
        print(f"Wrote {total_rows} REF/ALT expression rows to {args.out}")
        return

    pairs = load_sequence_pairs(args.input)
    output_rows = run_inference_on_pairs(
        pairs=pairs,
        model=model,
        dna_tokenizer=dna_tokenizer,
        text_tokenizer=text_tokenizer,
        description=description_text,
        description_name=args.description_name,
        dna_max_seq_len=int(config["args_params"]["input_seq_len"]),
        text_max_seq_len=int(config["shared_dataset_params"]["text_max_seq_len"]),
        sequence_batch_size=args.batch_size,
        device=device,
        use_amp=not args.no_amp,
    )
    write_table(args.out, output_rows)
    print(f"Wrote {len(output_rows)} REF/ALT expression rows to {args.out}")


if __name__ == "__main__":
    main()
