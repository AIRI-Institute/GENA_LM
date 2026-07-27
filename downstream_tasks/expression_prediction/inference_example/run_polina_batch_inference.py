#!/usr/bin/env python

from pathlib import Path
import argparse
import os
import sys

import pandas as pd
import torch
import torch._dynamo
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GENA_LM expression inference for valid/test genes."
    )

    parser.add_argument(
        "--task-root",
        default="/home/jovyan/dpanc/benchmarking/GENA_LM",
        help="Task folder with models, json_runs, outputs, cache",
    )
    parser.add_argument(
        "--gena-home",
        default="/home/jovyan/dpanc/GENA_LM/GENA_LM_expression_branch",
        help="GENA_LM repo root",
    )
    parser.add_argument(
        "--data-root",
        default="/home/jovyan/dpanc/benchmarking/data",
        help="Folder with human.valid/test.forward/reverse.csv and hg38.fna",
    )
    parser.add_argument(
        "--experiment-config",
        default=None,
        help="Path to inference.yaml. Default: <gena-home>/downstream_tasks/expression_prediction/inference_example/inference.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to pytorch_model.bin. Default: <task-root>/models/full_model/pytorch_model.bin",
    )
    parser.add_argument(
        "--json-dir",
        default=None,
        help="Folder with cell JSON files. Default: <task-root>/json_runs/json_14",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="Use human.<split>.forward/reverse.csv",
    )
    parser.add_argument("--forward-intervals", default=None)
    parser.add_argument("--reverse-intervals", default=None)
    parser.add_argument("--genome", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-before", type=int, default=512)
    parser.add_argument("--token-len-for-fetch", type=int, default=15)
    parser.add_argument("--dna-tokenizer", default=None)
    parser.add_argument("--text-tokenizer", default=None)
    parser.add_argument("--dna-max-seq-len", type=int, default=None)
    parser.add_argument("--text-max-seq-len", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check paths/imports and prepare inputs; do not load checkpoint or run model",
    )

    return parser.parse_args()


def make_cache_dirs(task_root):
    cache_root = task_root / "cache"
    os.environ["TMPDIR"] = str(cache_root / "tmp")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    for key in ["TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    torch._dynamo.config.suppress_errors = True


def main():
    args = parse_args()

    task_root = Path(args.task_root)
    gena_home = Path(args.gena_home)
    data_root = Path(args.data_root)

    experiment_config_path = Path(args.experiment_config) if args.experiment_config else (
        gena_home / "downstream_tasks/expression_prediction/inference_example/inference.yaml"
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        task_root / "models/full_model/pytorch_model.bin"
    )
    json_dir = Path(args.json_dir) if args.json_dir else (task_root / "json_runs/json_14")
    forward_intervals = Path(args.forward_intervals) if args.forward_intervals else (
        data_root / f"human.{args.split}.forward.csv"
    )
    reverse_intervals = Path(args.reverse_intervals) if args.reverse_intervals else (
        data_root / f"human.{args.split}.reverse.csv"
    )
    genome_path = Path(args.genome) if args.genome else (data_root / "hg38.fna")
    output_path = Path(args.output) if args.output else (
        task_root / "outputs" / f"gena_lm_{args.split}_{json_dir.name}_predictions.csv"
    )

    make_cache_dirs(task_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["GENALM_HOME"] = str(gena_home)
    sys.path.insert(0, str(gena_home))

    from downstream_tasks.expression_prediction.expression_model_final import ExpressionCounts
    from downstream_tasks.expression_prediction.inference_example.inference_input_utils import (
        prepare_inference_inputs_from_intervals,
    )

    print("task_root:", task_root)
    print("gena_home:", gena_home)
    print("experiment_config:", experiment_config_path)
    print("checkpoint:", checkpoint_path)
    print("json_dir:", json_dir)
    print("forward:", forward_intervals)
    print("reverse:", reverse_intervals)
    print("genome:", genome_path)
    print("output:", output_path)
    print("batch_size:", args.batch_size)
    print("device:", args.device)

    required_paths = [
        gena_home,
        experiment_config_path,
        json_dir,
        forward_intervals,
        reverse_intervals,
        genome_path,
    ]
    if not args.dry_run:
        required_paths.append(checkpoint_path)

    missing = [str(path) for path in required_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing paths:\n" + "\n".join(missing))

    with initialize_config_dir(str(experiment_config_path.parent), version_base=None):
        experiment_config = compose(config_name=experiment_config_path.name)

    model_kwargs = instantiate(experiment_config["model_kwargs"])

    dna_tokenizer_name = args.dna_tokenizer or experiment_config["args_params"]["gen_tokenizer"]
    text_tokenizer_name = args.text_tokenizer or experiment_config["shared_dataset_params"]["text_tokenizer"]
    dna_tokenizer = AutoTokenizer.from_pretrained(dna_tokenizer_name)
    text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name, padding_side="left")

    dna_max_seq_len = args.dna_max_seq_len or int(experiment_config["args_params"]["input_seq_len"])
    text_max_seq_len = args.text_max_seq_len or int(
        experiment_config["shared_dataset_params"]["text_max_seq_len"]
    )

    prepared = prepare_inference_inputs_from_intervals(
        json_dir=str(json_dir),
        forward_intervals_path=str(forward_intervals),
        reverse_intervals_path=str(reverse_intervals),
        genome_path=str(genome_path),
        gen_tokenizer=dna_tokenizer,
        text_tokenizer=text_tokenizer,
        gen_max_seq_len=dna_max_seq_len,
        text_max_seq_len=text_max_seq_len,
        cache_dir=str(task_root / "cache/inference"),
        num_before=int(dna_max_seq_len // 2),
        token_len_for_fetch=args.token_len_for_fetch,
    )

    genes = prepared["genes"]
    experiments = prepared["experiments"]
    tokenized_dna = prepared["tokenized_DNA"]
    tokenized_descriptions = prepared["tokenized_descriptions"]

    print("genes:", len(genes))
    print("cell JSONs:", len(experiments))
    print("experiments:", list(experiments.keys())[:10], "..." if len(experiments) > 10 else "")
    print("DNA input shape:", tuple(tokenized_dna["input_ids"].shape))
    print("gene caches:", prepared["gene_cache_paths"])
    print("description cache:", prepared["description_cache_path"])

    if args.dry_run:
        print("dry-run finished; model was not loaded")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(device))

    model = ExpressionCounts(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    input_ids = tokenized_dna["input_ids"].to(device)
    attention_mask = tokenized_dna["attention_mask"].to(device)

    pred_dict = {}

    for experiment in experiments:
        print("experiment:", experiment)

        desc_input_ids_all = tokenized_descriptions[experiment]["input_ids"].to(device)
        desc_attention_mask_all = tokenized_descriptions[experiment]["attention_mask"].to(device)

        experiment_outputs = []

        for start in range(0, input_ids.shape[0], args.batch_size):
            end = min(start + args.batch_size, input_ids.shape[0])

            batch_input_ids = input_ids[start:end]
            batch_attention_mask = attention_mask[start:end]
            batch_desc_input_ids = desc_input_ids_all[start:end].unsqueeze(1)
            batch_desc_attention_mask = desc_attention_mask_all[start:end].unsqueeze(1)

            dataset_flag = torch.zeros(
                size=(batch_input_ids.shape[0], 1),
                device=device,
                dtype=torch.bool,
            )

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16), torch.no_grad():
                output = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    desc_input_ids=batch_desc_input_ids,
                    desc_attention_mask=batch_desc_attention_mask,
                    dataset_flag=dataset_flag,
                )

            experiment_outputs.append(output["logits"][:, 0, 0].detach().float().cpu())

            if start == 0 or start % (args.batch_size * 10) == 0:
                print(f"  processed {end}/{input_ids.shape[0]}")

        pred_dict[experiment] = torch.cat(experiment_outputs, dim=0).float().numpy()

    gene_ids = [record["gene_id"] for record in genes.values()]
    pred_df = pd.DataFrame(pred_dict)
    pred_df.insert(0, "gene_id", gene_ids)
    pred_df.to_csv(output_path, index=False)

    print("saved:", output_path)
    print("shape:", pred_df.shape)
    print(pred_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
