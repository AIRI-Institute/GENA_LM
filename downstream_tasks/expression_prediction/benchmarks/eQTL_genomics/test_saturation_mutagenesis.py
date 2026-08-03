"""Focused tests for the compact saturation-mutagenesis pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parents[1]
for path in (TASK_DIR, TASK_DIR / "api" / "src", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gena_expression.sequences import AnnotatedSequence, Feature
from gena_expression.inference.tokenization import CenteredTokenizer
from expression_dataset_final import ExpressionDataset
from saturation_mutagenesis import (
    BASE_TO_CODE,
    TSSPlan,
    TSSRecord,
    alternatives,
    initialize_shard,
    iter_variant_scores,
    load_catalog,
    merge_shards,
    mutation_batches,
    render_description,
    validate_checkpoint_config,
)


def record(name: str = "tss-1", strand: str = "+") -> TSSRecord:
    return TSSRecord(name, "gene-1", "GENE", "chr1", 100, strand, "test")


def plan(name: str = "tss-1", strand: str = "+") -> TSSPlan:
    sequence = AnnotatedSequence(
        "NNACGTNN",
        name=name,
        features=(Feature("tss", 4, 5, type="tss", strand=strand),),
    )
    return TSSPlan(
        record=record(name, strand),
        sequence=sequence,
        center=4,
        mutation_start=2,
        mutation_end=6,
        genomic_start_0based=98,
        genomic_end_0based=102,
        ref_codes=np.asarray([BASE_TO_CODE[x] for x in "ACGT"], dtype=np.uint8),
        position_offsets=np.arange(4, dtype=np.int32),
    )


def provenance(expected: int = 1) -> dict[str, object]:
    return {"schema_version": "2", "expected_tss_count": expected, "catalog_sha256": "abc"}


def test_alternatives_are_lexicographic() -> None:
    assert alternatives("A") == ("C", "G", "T")
    assert alternatives("G") == ("A", "C", "T")


def test_catalog_converts_one_based_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "catalog.tsv"
    columns = ["tss_id", "gene_id", "gene_name", "chromosome", "tss_position_1based", "strand", "annotation_source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerow(dict(zip(columns, ["x", "g", "G", "chr1", 101, "-", "source"])))
    item = load_catalog(path)[0]
    assert item.tss_0based == 100
    assert item.strand == "-"


def test_description_uses_expression_dataset_formatter(tmp_path: Path) -> None:
    path = tmp_path / "description.json"
    path.write_text('{"Characteristics[cell_type]":"K_562", "Parameter dose":"\\\"high_value\\\""}', encoding="utf-8")
    metadata, text = render_description(path)
    assert text == ExpressionDataset.make_description_from_json(
        metadata, path.stem, str(path)
    )
    assert text == "Characteristics[cell type] is K 562. Parameter dose is high value."


def _write_model_config(path: Path, *, weight: int = 8, encoder: str = "encoder") -> None:
    path.write_text(
        "model_kwargs:\n"
        "  _target_: builtins.dict\n"
        f"  hf_model_name: {encoder}\n"
        "  hf_model_name_decoder: decoder\n"
        f"  weight: {weight}\n"
        "  config:\n"
        "    _target_: builtins.dict\n",
        encoding="utf-8",
    )


def test_checkpoint_requires_exactly_one_sibling_yaml(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"checkpoint")
    inference = tmp_path / "inference.yml"
    _write_model_config(inference)
    with pytest.raises(ValueError, match="exactly one"):
        validate_checkpoint_config(checkpoint, inference)
    _write_model_config(tmp_path / "first.yaml")
    _write_model_config(tmp_path / "second.yaml")
    with pytest.raises(ValueError, match="found 2"):
        validate_checkpoint_config(checkpoint, inference)


def test_checkpoint_config_matches_except_model_asset_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"checkpoint")
    inference = tmp_path / "inference.yml"
    checkpoint_config = tmp_path / "training.yaml"
    _write_model_config(inference, encoder="inference-encoder")
    _write_model_config(checkpoint_config, encoder="checkpoint-encoder")
    assert validate_checkpoint_config(checkpoint, inference) == checkpoint_config
    _write_model_config(checkpoint_config, weight=5, encoder="checkpoint-encoder")
    with pytest.raises(ValueError, match=r"model_kwargs.weight.*inference=8, checkpoint=5"):
        validate_checkpoint_config(checkpoint, inference)


def test_mutation_batches_retokenize_three_alternatives() -> None:
    batches = list(mutation_batches(plan(), batch_size=5))
    assert sum(len(items) for _, items in batches) == 12
    first = batches[0][1][:3]
    assert [item.sequence[2] for item in first] == ["C", "G", "T"]
    assert all(item.features[0].name == "tss" for item in first)


def test_collapsed_n_gap_keeps_source_coordinates_after_token_selection() -> None:
    """A selected leading gap must retain the entire N-run, not one '-' bp."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HERE.parents[3] / "data/tokenizers/t2t_1000h_multi_32k")
    centered = CenteredTokenizer(tokenizer, dna_max_seq_len=6, token_len_for_fetch=20, num_before=2)
    cases = (
        ("+", "A" * 100 + "N" * 12 + "C" + "G" * 100, 113, (100, 112)),
        ("-", "A" * 100 + "C" + "N" * 12 + "G" * 100, 101, (101, 113)),
    )
    for strand, text, center, expected_span in cases:
        sequence = AnnotatedSequence(
            text,
            features=(Feature("tss", center, center + 1, type="tss"),),
        )
        result = centered.tokenize(sequence, center="tss", strand=strand)
        gap = next(row for row in result.tokens if row["token_id"] == 5)
        assert gap["token"] == "-"
        assert (gap["start"], gap["end"], gap["length"]) == (*expected_span, 12)
        assert sequence.sequence[gap["start"] : gap["end"]] == "N" * 12
        assert all(left["end"] == right["start"] for left, right in zip(result.tokens, result.tokens[1:]))


def test_hdf5_roundtrip_and_merge(tmp_path: Path) -> None:
    first_path = tmp_path / "shard-0000-of-0002.h5"
    second_path = tmp_path / "shard-0001-of-0002.h5"
    first_plan, second_plan = plan("plus", "+"), plan("minus", "-")
    initialize_shard(first_path, [first_plan], provenance(2), 0, 2)
    initialize_shard(second_path, [second_plan], provenance(2), 1, 2)
    for path in (first_path, second_path):
        with h5py.File(path, "r+") as handle:
            absolute = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
            handle["variant_atac_sum"][:] = absolute
            handle["tss/reference_atac_sum"][:] = [1.5, 2.5]
            handle["scores"][:] = absolute - np.asarray([1.5, 2.5], dtype=np.float32)
            handle["tss/status"][:] = 1
    final = tmp_path / "final.h5"
    merge_shards([first_path, second_path], final)
    rows = list(iter_variant_scores(final, "minus"))
    assert len(rows) == 12
    assert rows[0]["position_1based"] == 99
    assert rows[-1]["position_1based"] == 102
    assert rows[0]["ref"] == "A"
    assert rows[0]["forward_reference_atac_sum"] == 1.5
    assert rows[0]["reverse_complement_reference_atac_sum"] == 2.5
    assert rows[0]["forward_variant_atac_sum"] == 0.0
    assert rows[0]["forward_delta"] == -1.5
