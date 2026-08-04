#!/usr/bin/env python3
"""Score each explicit catalog variant in a variant-centered genomic context."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[3]
API_SRC = TASK_DIR / "api" / "src"
for path in (REPO_ROOT, TASK_DIR, API_SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from saturation_mutagenesis import (  # noqa: E402
    _load_checkpoint_runtime_config,
    dna_token_sides,
    find_checkpoint_config,
    render_description,
    sha256_file,
)


SCHEMA_VERSION = "variant-catalog-1"
VARIANT_TYPES = {"SNV", "insertion", "deletion"}


@dataclass(frozen=True)
class CatalogVariant:
    variant_id: str
    chromosome: str
    position_1based: int
    reference: str
    alternate: str
    variant_type: str
    present_in_train: bool
    present_in_validation: bool
    train_signal_count: int
    validation_signal_count: int
    total_signal_count: int

    @property
    def position_0based(self) -> int:
        return self.position_1based - 1


@dataclass(frozen=True)
class CatalogGenomeContext:
    """Build a variant-centered pair whose reference is the catalog REF allele."""

    length: int

    def build(self, variant: Any, *, genome: Any = None, name: str | None = None) -> Any:
        from gena_expression.sequences import SequencePair

        if genome is None or variant.chrom is None or variant.pos is None:
            raise ValueError("CatalogGenomeContext requires a genome and genomic variant")
        start = variant.pos - self.length // 2
        local_start = variant.pos - start
        reference = genome.sequence(
            variant.chrom,
            start,
            start + self.length,
            strand="+",
            include_features=True,
            name=name or variant.id or "reference",
        )
        # Some catalog REF alleles are not the hg38 allele. Build the literal
        # requested REF and ALT sequences so every row retains its stated sign.
        reference = reference.replace(
            local_start,
            local_start + len(variant.ref),
            variant.ref,
            preserve_partial_features=True,
        ).remove_features(type="replacement")
        reference = reference.add_feature(
            "variant",
            local_start,
            local_start + max(1, len(variant.ref)),
            type="variant",
            source="variant_catalog",
            metadata=variant.to_dict(),
        )
        alternate = variant.apply_to(reference, offset=local_start)
        pair = SequencePair(
            ref=reference,
            alt=alternate,
            variant=variant,
            metadata={"context": "CatalogGenomeContext", "length": self.length},
        )
        pair.assert_compatible()
        return pair


def _boolean(value: str, *, field: str, variant_id: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid {field} for {variant_id}: {value!r}")


def load_variant_catalog(path: str | Path) -> list[CatalogVariant]:
    """Validate and load the supplied 1-based hg38 variant table."""

    required = {
        "variant_id", "chromosome", "position_1based", "reference", "alternate",
        "variant_type", "present_in_train", "present_in_validation",
        "train_signal_count", "validation_signal_count", "total_signal_count",
    }
    variants: list[CatalogVariant] = []
    seen: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Variant catalog is missing: {', '.join(sorted(missing))}")
        for row in reader:
            variant_id = row["variant_id"]
            if variant_id in seen:
                raise ValueError(f"Duplicate variant_id: {variant_id}")
            seen.add(variant_id)
            chromosome = row["chromosome"]
            position = int(row["position_1based"])
            reference = row["reference"].upper()
            alternate = row["alternate"].upper()
            variant_type = row["variant_type"]
            expected_id = f"{chromosome}-{position}-{reference}-{alternate}"
            if position < 1 or not chromosome or not reference or not alternate:
                raise ValueError(f"Invalid variant row: {variant_id}")
            if variant_id != expected_id:
                raise ValueError(f"variant_id mismatch: {variant_id!r} != {expected_id!r}")
            if variant_type not in VARIANT_TYPES:
                raise ValueError(f"Invalid variant_type for {variant_id}: {variant_type!r}")
            if variant_type == "SNV" and (
                len(reference) != 1 or len(alternate) != 1 or reference == alternate
            ):
                raise ValueError(f"Invalid SNV alleles for {variant_id}")
            if variant_type == "insertion" and len(alternate) <= len(reference):
                raise ValueError(f"Invalid insertion alleles for {variant_id}")
            if variant_type == "deletion" and len(reference) <= len(alternate):
                raise ValueError(f"Invalid deletion alleles for {variant_id}")
            if set(reference + alternate).difference("ACGT"):
                raise ValueError(f"Non-ACGT allele for {variant_id}")
            train_count = int(row["train_signal_count"])
            validation_count = int(row["validation_signal_count"])
            total_count = int(row["total_signal_count"])
            if min(train_count, validation_count, total_count) < 0:
                raise ValueError(f"Negative signal count for {variant_id}")
            if train_count + validation_count != total_count:
                raise ValueError(f"Signal counts do not add up for {variant_id}")
            variants.append(
                CatalogVariant(
                    variant_id=variant_id,
                    chromosome=chromosome,
                    position_1based=position,
                    reference=reference,
                    alternate=alternate,
                    variant_type=variant_type,
                    present_in_train=_boolean(
                        row["present_in_train"], field="present_in_train", variant_id=variant_id
                    ),
                    present_in_validation=_boolean(
                        row["present_in_validation"],
                        field="present_in_validation",
                        variant_id=variant_id,
                    ),
                    train_signal_count=train_count,
                    validation_signal_count=validation_count,
                    total_signal_count=total_count,
                )
            )
    return variants


def _config_defaults(path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    values = {
        key: OmegaConf.select(config, key)
        for key in ("checkpoint", "genome_fasta", "description_json", "score_window_bp")
    }
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    return values


def _local_path(value: Any, *, config_path: Path, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = config_path.parent / path
    path = path.absolute()
    anchored = path.parent.resolve() / path.name
    try:
        anchored.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"Configured {name} must be inside {REPO_ROOT}: {path}") from error
    return path


def _add_shared(parser: argparse.ArgumentParser) -> None:
    config_path = HERE / "variant_inference_config.yaml"
    defaults = _config_defaults(config_path)
    parser.add_argument("--catalog", type=Path, default=HERE / "data" / "variant_catalog.tsv")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_local_path(defaults["checkpoint"], config_path=config_path, name="checkpoint"),
    )
    parser.add_argument(
        "--genome-fasta",
        type=Path,
        default=_local_path(defaults["genome_fasta"], config_path=config_path, name="genome_fasta"),
    )
    parser.add_argument(
        "--description-json",
        type=Path,
        default=_local_path(
            defaults["description_json"], config_path=config_path, name="description_json"
        ),
    )
    parser.add_argument("--score-window-bp", type=int, default=int(defaults["score_window_bp"]))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--preprocessing-workers", type=int, default=10)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--fetch-bp-per-token", type=int, default=20)
    parser.add_argument("--limit", type=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pilot = commands.add_parser("pilot")
    _add_shared(pilot)
    pilot.add_argument("--device", default="cuda:0")
    pilot.set_defaults(limit=10, shard_index=0, shard_count=1)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    _add_shared(worker)
    worker.add_argument("--device", required=True)
    worker.add_argument("--shard-index", type=int, required=True)
    worker.add_argument("--shard-count", type=int, required=True)
    run = commands.add_parser("run")
    _add_shared(run)
    run.add_argument("--devices", default="0")
    merge = commands.add_parser("merge")
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _initialize_shard(
    path: Path,
    rows: Sequence[tuple[int, CatalogVariant]],
    attrs: dict[str, Any],
) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(attrs)
        handle.create_dataset("catalog_index", data=[index for index, _ in rows], dtype=np.int64)
        for name in ("variant_id", "chromosome", "reference", "alternate", "variant_type"):
            handle.create_dataset(
                name,
                data=[getattr(row, name) for _, row in rows],
                dtype=_string_dtype(),
            )
        handle.create_dataset(
            "position_1based",
            data=[row.position_1based for _, row in rows],
            dtype=np.int64,
        )
        handle.create_dataset(
            "present_in_train",
            data=[row.present_in_train for _, row in rows],
            dtype=np.bool_,
        )
        handle.create_dataset(
            "present_in_validation",
            data=[row.present_in_validation for _, row in rows],
            dtype=np.bool_,
        )
        for name in ("train_signal_count", "validation_signal_count", "total_signal_count"):
            handle.create_dataset(
                name,
                data=[getattr(row, name) for _, row in rows],
                dtype=np.int32,
            )
        handle.create_dataset("score", data=np.full(len(rows), np.nan, dtype=np.float32))
        handle.create_dataset("status", data=np.zeros(len(rows), dtype=np.uint8))
        handle.create_dataset("error", data=["" for _ in rows], dtype=_string_dtype())
        handle.flush()
    os.replace(temporary, path)


def _chunks(values: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def run_worker(args: argparse.Namespace) -> None:
    from gena_expression.conditions import Condition
    from gena_expression.config import SCALAR_RETENTION
    from gena_expression.dna import Genome
    from gena_expression.inference import SequenceModel
    from gena_expression.scoring import TrackWindowScorer, VariantInterpreter
    from gena_expression.variants import Variant

    all_variants = load_variant_catalog(args.catalog)
    if args.limit is not None:
        all_variants = all_variants[: args.limit]
    rows = [
        (index, variant)
        for index, variant in enumerate(all_variants)
        if index % args.shard_count == args.shard_index
    ]
    checkpoint_config = find_checkpoint_config(args.checkpoint)
    runtime = _load_checkpoint_runtime_config(checkpoint_config)
    total_tokens = int(runtime["model_input_seq_len"])
    upstream_tokens, downstream_tokens = dna_token_sides(total_tokens, (total_tokens - 2) // 2)
    context_length = 2 * max(upstream_tokens, downstream_tokens) * args.fetch_bp_per_token
    variant_center = context_length // 2
    _, description = render_description(args.description_json)
    attrs = {
        "schema_version": SCHEMA_VERSION,
        "catalog_sha256": sha256_file(args.catalog),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_config_sha256": sha256_file(checkpoint_config),
        "genome_fai_sha256": sha256_file(f"{args.genome_fasta}.fai"),
        "description_sha256": sha256_file(args.description_json),
        "total_variants": len(all_variants),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "dna_input_seq_len": total_tokens,
        "dna_tokens_upstream": upstream_tokens,
        "dna_tokens_downstream": downstream_tokens,
        "context_length_bp": context_length,
        "variant_center_local": variant_center,
        "score_window": f"[variant-{args.score_window_bp},variant+{args.score_window_bp + 1})",
        "score_definition": "alternative ATAC sum - reference ATAC sum",
    }
    shard_path = args.output_dir / f"variant-shard-{args.shard_index:04d}-of-{args.shard_count:04d}.h5"
    _initialize_shard(shard_path, rows, attrs)

    genome = Genome(args.genome_fasta, build="hg38")
    os.environ.setdefault("CAGI5_MODEL_ROOT", str(args.checkpoint.resolve().parents[1]))
    model = SequenceModel.load(
        model_cls=str(runtime["model_class"]),
        checkpoint=args.checkpoint,
        config=checkpoint_config,
        dna_tokenizer=runtime["dna_tokenizer"],
        description_tokenizer=runtime["description_tokenizer"],
        dna_max_seq_len=total_tokens,
        num_before=upstream_tokens,
        desc_max_seq_len=int(runtime["text_max_seq_len"]),
        token_len_for_fetch=args.fetch_bp_per_token,
        device=args.device,
    )
    context = CatalogGenomeContext(length=context_length)
    condition = Condition(name=args.description_json.stem, description=description)
    scorer = TrackWindowScorer(
        track="atac",
        center=variant_center,
        left_bp=args.score_window_bp,
        right_bp=args.score_window_bp + 1,
        aggregate="sum",
        sign="alt-ref",
        channel=0,
    )
    interpreter = VariantInterpreter(model)
    started = time.monotonic()
    completed = 0
    with h5py.File(shard_path, "r+") as output:
        pending = np.flatnonzero(output["status"][:] == 0).tolist()
        for local_indices in _chunks(pending, args.batch_size):
            api_variants = []
            for local_index in local_indices:
                row = rows[local_index][1]
                api_variants.append(
                    Variant(
                        chrom=row.chromosome, pos=row.position_0based,
                        ref=row.reference, alt=row.alternate, id=row.variant_id,
                    )
                )
            results = interpreter.score_variants(
                api_variants,
                context=context,
                condition=condition,
                scorer=scorer,
                genome=genome,
                center=variant_center,
                grouping="condition",
                pair_execution="joint",
                preprocessing_workers=args.preprocessing_workers,
                max_pairs_per_forward=args.batch_size,
                prefetch_batches=args.prefetch_batches,
                show_progress=False,
                retention=SCALAR_RETENTION,
            )
            scores = np.asarray([float(result.score) for result in results], dtype=np.float32)
            if len(scores) != len(local_indices) or not np.all(np.isfinite(scores)):
                raise ValueError("Variant batch returned missing or non-finite scores")
            output["score"][local_indices] = scores
            output["status"][local_indices] = 1
            output.flush()
            completed += len(local_indices)
            elapsed = time.monotonic() - started
            print(
                f"shard={args.shard_index} completed={completed}/{len(pending)} "
                f"variants_per_second={completed / elapsed:.2f}",
                flush=True,
            )


def launch(args: argparse.Namespace) -> None:
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one GPU")
    processes = []
    for shard_index, device in enumerate(devices):
        command = [sys.executable, str(Path(__file__).resolve()), "worker"]
        for key, value in vars(args).items():
            if key in {"command", "devices"} or value is None:
                continue
            command.extend(("--" + key.replace("_", "-"), str(value)))
        command.extend(
            (
                "--device",
                f"cuda:{device}",
                "--shard-index",
                str(shard_index),
                "--shard-count",
                str(len(devices)),
            )
        )
        processes.append(subprocess.Popen(command))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise SystemExit(f"Worker exit codes: {codes}")


def merge_shards(paths: Sequence[Path], output: Path) -> None:
    if not paths:
        raise ValueError("No variant shards found")
    handles = [h5py.File(path, "r") for path in paths]
    try:
        first_attrs = dict(handles[0].attrs)
        comparable = {key: value for key, value in first_attrs.items() if key != "shard_index"}
        for handle in handles:
            current = {key: value for key, value in handle.attrs.items() if key != "shard_index"}
            if current != comparable:
                raise ValueError(f"Incompatible shard provenance: {handle.filename}")
            if not np.all(handle["status"][:] == 1):
                raise ValueError(f"Incomplete shard: {handle.filename}")
        indices = np.concatenate([handle["catalog_index"][:] for handle in handles])
        expected = int(first_attrs["total_variants"])
        if len(indices) != expected or set(indices.tolist()) != set(range(expected)):
            raise ValueError("Merged shards have missing or duplicate catalog rows")
        order = np.argsort(indices)
        temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(temporary, "w") as target:
            for key, value in comparable.items():
                target.attrs[key] = value
            for name in handles[0].keys():
                values = np.concatenate([handle[name][:] for handle in handles])[order]
                target.create_dataset(name, data=values, dtype=handles[0][name].dtype)
            target.flush()
        os.replace(temporary, output)
    finally:
        for handle in handles:
            handle.close()


def main() -> None:
    args = parse_args()
    if args.command in {"pilot", "worker"}:
        run_worker(args)
    elif args.command == "run":
        launch(args)
    else:
        merge_shards(sorted(args.output_dir.glob("variant-shard-*.h5")), args.output)


if __name__ == "__main__":
    main()
