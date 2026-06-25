import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

DEFAULT_CAGI5_HOME = "/workspace-SR003.nfs2/estsoi/CAGI5_benchmark"
DEFAULT_GENA_LM_PATH = (
    f"{DEFAULT_CAGI5_HOME}/GENA_LM/downstream_tasks/expression_prediction"
)

os.environ["GENALM_HOME"] = DEFAULT_CAGI5_HOME
os.environ["PYTHONPATH"] = os.pathsep.join(
    part for part in [os.environ.get("PYTHONPATH", ""), DEFAULT_GENA_LM_PATH] if part
)
if DEFAULT_GENA_LM_PATH not in sys.path:
    sys.path.append(DEFAULT_GENA_LM_PATH)

from DescriptionLookup import DescriptionLookup
from contexts import GenomeRegion
from contexts import Plasmid
from contexts import infer_table18_primer_tails
from CAGI5_bench import CAGI5_bench


ELEMENTS = [
    "F9",
    "FOXE1",
    "GP1BA",
    "HBB",
    "HBG1",
    "HNF4A",
    "LDLR",
    "LDLR.2",
    "MSMB",
    "PKLR-24h",
    "PKLR-48h",
    "TERT-GAa",
    "TERT-GBM",
    "TERT-GSc",
    "TERT-HEK",
]

DESCRIPTION_KEYS = [
    "HEK293T",
    "HEL92.1.7",
    "HaCaT",
    "HeLa",
    "HepG2",
    "K562",
    "LNCaP",
    "MIN6",
    "NIH-3T3",
    "Neuro-2a",
    "SF7996",
    "SK-MEL-28",
]


def corr_to_float(value: Any) -> float:
    """Convert bench.corr[element] to the scalar true-vs-pred correlation."""
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        if value.ndim >= 2:
            value = value[0, 1]
        else:
            value = value.reshape(-1)[0]
        return float(value.item())
    return float(value)


def run_experiment(
    bench: CAGI5_bench,
    *,
    name: str,
    batch_size: int,
    n_workers: int,
    initialize_dataset_kwargs: dict[str, Any],
    reload_model: bool = False,
) -> dict[str, Any]:
    print(f"\n=== {name} ===", flush=True)
    bench.initialize_dataset(**initialize_dataset_kwargs)
    bench.make_dataloader(batch_size=batch_size, n_workers=n_workers)
    if reload_model:
        bench.initialize_model()
    bench.run_inference()

    row = {"experiment": name}
    for element in ELEMENTS:
        corr_value = bench.corr.get(element)
        row[element] = corr_to_float(corr_value) if corr_value is not None else math.nan
    print(row)
    return row


def write_table(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", *ELEMENTS]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    
    cagi5_home = DEFAULT_CAGI5_HOME
    
    parser = argparse.ArgumentParser(
        description="Run CAGI5 benchmark experiments and write an experiment x promoter-element correlation table."
    )
    parser.add_argument("--output", default="cagi5_promoter_corr_table_2.csv")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--reload-model-each-experiment", action="store_true")
    parser.add_argument('--cls', default="downstream_tasks.expression_prediction.expression_model_final.ExpressionCounts")
    parser.add_argument('--checkpoint', default=f"{cagi5_home}/models/expression_model/pytorch_model.bin")
    parser.add_argument('--model_config', default=f"{cagi5_home}/GENA_LM/downstream_tasks/expression_prediction/configs/final.yaml")
    args = parser.parse_args()

    

    lookup = DescriptionLookup(
        json_dir=f"{cagi5_home}/descriptions",
        keys=DESCRIPTION_KEYS,
    )

    plasmid = Plasmid(
        "https://www.ncbi.nlm.nih.gov/nuccore/MK484103.1",
        data_storage=f"{cagi5_home}/plasmids",
        reporter_feature_name="similar to firefly luciferase; synthetic luc2 version of the luciferase gene",
    )

    hg38_path = "/home/jovyan/.cache/mpramnist/data/Kircher/hg38.fa"
    tails = infer_table18_primer_tails(hg38_path, flank=200, min_match=15)
    plasmid_context_no_aug = plasmid.plasmid_context(
        primer_tails=tails,
        context_size=None,
        circular=True,
    )
    plasmid_context_aug = plasmid.plasmid_context(
        primer_tails=tails,
        context_size=20_000,
        circular=True,
    )

    genome = GenomeRegion(genome_path=hg38_path)
    genome_context = genome.genomic_context(region="AAVS1", target_length=20_000)

    bench = CAGI5_bench(
        model_cls=args.cls,
        model_checkpoint=args.checkpoint,
        model_config=args.model_config,
        dna_tokenizer=f"{cagi5_home}/GENA_LM/data/tokenizers/t2t_1000h_multi_32k",
        desc_max_seq_len=1024,
        token_len_for_fetch=100,
        desc_tokenizer="Qwen/Qwen3-Embedding-0.6B",
        dna_max_seq_len=1024,
        num_before=511,
        celltype2desc=lookup,
        device=args.device,
    )
    bench.initialize_model()

    common_custom = {
        "elements": ELEMENTS,
        "mode": "custom",
        "min_tags": 10,
    }

    experiments = [
        {
            "name": "Only plasmid context, less then 1022 tokens. In this case, bs > 1 is impossible",
            "batch_size": 1,
            "n_workers": 8,
            "initialize_dataset_kwargs": {
                **common_custom,
                "context_fn": plasmid_context_no_aug,
            },
        },
        {
            "name": "Human genome safe harbor context",
            "batch_size": 1,
            "n_workers": 8,
            "initialize_dataset_kwargs": {
                **common_custom,
                "padding_fn": genome_context,
            },
        },
        {
            "name": "Plasmid context + genome safe harbor context",
            "batch_size": 1,
            "n_workers": 8,
            "initialize_dataset_kwargs": {
                **common_custom,
                "context_fn": plasmid_context_no_aug,
                "padding_fn": genome_context,
            },
        },
        {
            "name": "Plasmid context + plasmid duplications",
            "batch_size": 1,
            "n_workers": 8,
            "initialize_dataset_kwargs": {
                **common_custom,
                "context_fn": plasmid_context_aug,
            },
        },
        {
            "name": "Mpra fragments in their local genome context",
            "batch_size": 1,
            "n_workers": 5,
            "initialize_dataset_kwargs": {
                "elements": ELEMENTS,
                "mode": "default",
                "length": 20_000,
                "min_tags": 10,
            },
        },
        {
            "name": "MPRA fragment only, without context",
            "batch_size": 1,
            "n_workers": 5,
            "initialize_dataset_kwargs": {
                "elements": ELEMENTS,
                "mode": "custom",
                "min_tags": 10,
            },
        },
        {
            "name": "501 nucleotides around variant (AlphaGenome style)",
            "batch_size": 1,
            "n_workers": 5,
            "initialize_dataset_kwargs": {
                "elements": ELEMENTS,
                "mode": "default",
                "length": 501,
                "min_tags": 10,
            },
        },
    ]

    rows = [
        run_experiment(
            bench,
            name=experiment["name"],
            batch_size=experiment["batch_size"],
            n_workers=experiment["n_workers"],
            initialize_dataset_kwargs=experiment["initialize_dataset_kwargs"],
            reload_model=args.reload_model_each_experiment,
        )
        for experiment in experiments
    ]

    out_path = Path(args.output)
    write_table(rows, out_path)
    print(f"\nSaved correlation table to {out_path.resolve()}")


if __name__ == "__main__":
    main()
