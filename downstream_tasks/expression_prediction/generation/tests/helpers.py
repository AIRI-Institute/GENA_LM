"""Shared synthetic configuration and file fixtures."""

from __future__ import annotations

from pathlib import Path

import yaml


def complete_config(root: Path) -> dict:
    """Return a small but structurally complete project configuration."""

    return {
        "project": {"name": "test_generation"},
        "model": {
            "genalm_home": None,
            "model_cls": "model.py::Model",
            "checkpoint": "model.bin",
            "config": "model.yaml",
            "dna_tokenizer": {"path": "dna_tokenizer"},
            "description_tokenizer": {"identifier": "description/model"},
            "dna_max_seq_len": 32,
            "desc_max_seq_len": 16,
            "token_len_for_fetch": 4,
            "num_before": 8,
            "device": "cpu",
            "output_names": None,
            "description_formatter": None,
        },
        "genome": {
            "fasta_path": "genome.fa",
            "annotation_path": "genes.gtf",
            "build": "test",
            "chrom_style": "auto",
            "cache": True,
        },
        "sampler": {
            "seq_len": 4,
            "tss_sampling_window": 10,
            "seed": 2,
            "transcript_feature_types": None,
            "include_features": False,
            "unique_sequences": True,
            "max_attempts": 20,
        },
        "plasmid": {
            "genbank_path": "plasmids/test.gb",
            "fasta_path": None,
            "reporter_feature_name": "luc2",
            "replacement_feature_name": "replace_me",
            "region_feature_name": "promoter",
            "tokenization_center_name": "reporter_tss",
            "circular": True,
            "window_size": "full",
        },
        "conditions": {
            "seed": 3,
            "strict": True,
            "search_fields": None,
            "on_target": {
                "directory": "on_target",
                "keys": ["ON"],
                "filters": {},
                "key_filters": {},
            },
            "off_target": {
                "directory": "off_target",
                "keys": ["OFF"],
                "filters": {},
                "key_filters": {},
            },
        },
        "scorer": {"type": "expression", "params": {}},
        "fitness": {
            "type": "min_on_max_off",
            "params": {"on_power": 1.0, "off_power": 2.0, "off_weight": 1.0},
        },
        "offspring": {
            "type": "sequential",
            "params": {
                "crossover_probability": 0.5,
                "mutations_per_child": 1,
                "random_seed": 4,
            },
        },
        "optimization": {
            "population_size": 3,
            "offspring_size": 2,
            "n_generations": 1,
            "selection_strategy": "top_k",
            "elite_count": 1,
            "tournament_size": 2,
            "offspring_generator_mode": "records",
            "random_seed": 5,
        },
        "runtime": {
            "evaluation_chunk_size": 2,
            "grouping": "condition",
            "preprocessing_workers": 0,
            "preprocessing_backend": "process",
            "max_records_per_forward": None,
            "prefetch_batches": 1,
            "show_progress": False,
        },
        "output": {"directory": "results/run", "allow_existing": False},
    }


def write_config(root: Path, data: dict | None = None) -> Path:
    """Write a test configuration and return its path."""

    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(data or complete_config(root), sort_keys=False), encoding="utf-8")
    return path


def write_plasmid(root: Path, *, duplicate_target: bool = False) -> tuple[Path, Path]:
    """Write a simple 20 bp plasmid with replacement and reporter features."""

    plasmids = root / "plasmids"
    plasmids.mkdir(parents=True, exist_ok=True)
    sequence = "AAAACCCCGGGGTTTTACGT"
    duplicate = (
        '     misc_feature    8..9\n                     /label="replace_me"\n'
        if duplicate_target
        else ""
    )
    genbank = plasmids / "test.gb"
    genbank.write_text(
        "LOCUS       test_plasmid       20 bp    DNA     circular\n"
        "FEATURES             Location/Qualifiers\n"
        "     misc_feature    4..7\n"
        "                     /label=\"replace_me\"\n"
        + duplicate
        + "     CDS             12..18\n"
        "                     /gene=\"luc2\"\n"
        "ORIGIN\n"
        "        1 aaaaccccgg ggttttacgt\n"
        "//\n",
        encoding="utf-8",
    )
    fasta = plasmids / "test.fa"
    fasta.write_text(f">test_plasmid\n{sequence}\n", encoding="utf-8")
    return genbank, fasta
