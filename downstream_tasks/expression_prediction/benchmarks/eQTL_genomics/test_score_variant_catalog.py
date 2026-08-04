from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import pytest

from score_variant_catalog import (
    CatalogGenomeContext,
    CatalogVariant,
    _initialize_shard,
    load_variant_catalog,
    merge_shards,
)


FIELDS = (
    "variant_id",
    "chromosome",
    "position_1based",
    "reference",
    "alternate",
    "variant_type",
    "present_in_train",
    "present_in_validation",
    "train_signal_count",
    "validation_signal_count",
    "total_signal_count",
)


def _write_catalog(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _row(ref: str, alt: str, variant_type: str) -> dict[str, object]:
    return {
        "variant_id": f"chr1-101-{ref}-{alt}",
        "chromosome": "chr1",
        "position_1based": 101,
        "reference": ref,
        "alternate": alt,
        "variant_type": variant_type,
        "present_in_train": "True",
        "present_in_validation": "False",
        "train_signal_count": 2,
        "validation_signal_count": 0,
        "total_signal_count": 2,
    }


def test_load_variant_catalog_accepts_all_supported_types(tmp_path: Path) -> None:
    catalog = tmp_path / "variants.tsv"
    rows = [_row("A", "C", "SNV"), _row("A", "AT", "insertion"), _row("AT", "A", "deletion")]
    rows[1]["position_1based"] = 102
    rows[1]["variant_id"] = "chr1-102-A-AT"
    rows[2]["position_1based"] = 103
    rows[2]["variant_id"] = "chr1-103-AT-A"
    _write_catalog(catalog, rows)

    loaded = load_variant_catalog(catalog)

    assert [row.variant_type for row in loaded] == ["SNV", "insertion", "deletion"]
    assert loaded[0].position_0based == 100
    assert loaded[0].present_in_train is True
    assert loaded[0].present_in_validation is False


@pytest.mark.parametrize(
    ("ref", "alt", "variant_type"),
    [("A", "AT", "deletion"), ("AT", "A", "insertion"), ("A", "N", "SNV")],
)
def test_load_variant_catalog_rejects_invalid_alleles(
    tmp_path: Path,
    ref: str,
    alt: str,
    variant_type: str,
) -> None:
    catalog = tmp_path / "variants.tsv"
    _write_catalog(catalog, [_row(ref, alt, variant_type)])

    with pytest.raises(ValueError):
        load_variant_catalog(catalog)


def _variant(index: int) -> CatalogVariant:
    return CatalogVariant(
        variant_id=f"chr1-{index + 1}-A-C",
        chromosome="chr1",
        position_1based=index + 1,
        reference="A",
        alternate="C",
        variant_type="SNV",
        present_in_train=False,
        present_in_validation=True,
        train_signal_count=0,
        validation_signal_count=1,
        total_signal_count=1,
    )


def test_shards_merge_in_catalog_order(tmp_path: Path) -> None:
    attrs = {"schema_version": "test", "total_variants": 3, "shard_count": 2}
    shard_zero = tmp_path / "variant-shard-0000-of-0002.h5"
    shard_one = tmp_path / "variant-shard-0001-of-0002.h5"
    _initialize_shard(shard_zero, [(0, _variant(0)), (2, _variant(2))], {**attrs, "shard_index": 0})
    _initialize_shard(shard_one, [(1, _variant(1))], {**attrs, "shard_index": 1})
    with h5py.File(shard_zero, "r+") as handle:
        handle["score"][:] = [0.5, 2.5]
        handle["status"][:] = 1
    with h5py.File(shard_one, "r+") as handle:
        handle["score"][:] = [1.5]
        handle["status"][:] = 1

    merged = tmp_path / "merged.h5"
    merge_shards([shard_zero, shard_one], merged)

    with h5py.File(merged) as handle:
        np.testing.assert_array_equal(handle["catalog_index"][:], [0, 1, 2])
        np.testing.assert_allclose(handle["score"][:], [0.5, 1.5, 2.5])


def test_catalog_context_uses_literal_ref_and_alt_alleles() -> None:
    from gena_expression.sequences import AnnotatedSequence
    from gena_expression.variants import Variant

    class Genome:
        @staticmethod
        def sequence(
            chromosome: str,
            start: int,
            end: int,
            **kwargs: object,
        ) -> AnnotatedSequence:
            return AnnotatedSequence("G" * (end - start))

    variant = Variant(chrom="chr1", pos=100, ref="AT", alt="A", id="literal")
    pair = CatalogGenomeContext(length=20).build(variant, genome=Genome())

    assert pair.ref.sequence[10:12] == "AT"
    assert pair.alt.sequence[10:11] == "A"
    assert len(pair.ref) == 20
    assert len(pair.alt) == 19
