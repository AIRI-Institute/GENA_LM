"""Compare inference settings on the same sampled promoters and descriptions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .conditions import Condition
from .config import PredictionRetention, RetentionPolicy
from .dna import Genome, PromoterSampler
from .inference import SequenceModel
from .sequences import AnnotatedSequence, SequencePair


@dataclass(frozen=True)
class _Run:
    """Inference options for one traversal of the fixed workload."""

    grouping: str = "no_grouping"
    batch_size: int = 16
    preprocessing_workers: int = 0
    preprocessing_backend: str = "process"
    prefetch_batches: int = 1
    reverse_order: bool = False
    pair_execution: str = "separate"
    model_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grouping not in {"serial", "no_grouping", "condition", "sequence", "auto"}:
            raise ValueError(f"Unknown grouping: {self.grouping!r}")
        if self.preprocessing_backend not in {"thread", "process"}:
            raise ValueError(f"Unknown preprocessing backend: {self.preprocessing_backend!r}")
        if self.pair_execution not in {"separate", "joint"}:
            raise ValueError(f"Unknown pair execution: {self.pair_execution!r}")
        for name, minimum in (("batch_size", 1), ("prefetch_batches", 0), ("preprocessing_workers", 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.reverse_order) is not bool or not isinstance(self.model_overrides, Mapping):
            raise ValueError("Expected a boolean reverse_order and an object for model_overrides")


_DEFAULT_RUNS = {
    "batched": _Run(),
    "by_condition": _Run(grouping="condition"),
    "by_sequence": _Run(grouping="sequence"),
    "auto_reversed": _Run(grouping="auto", batch_size=4, reverse_order=True),
}
_DEFAULT_PAIR_RUNS = {
    "separate": _Run(),
    "joint": _Run(pair_execution="joint"),
}
_SERIAL = _Run(grouping="serial", batch_size=1, prefetch_batches=0)


def _take_promoters(
    sampler: Iterable[AnnotatedSequence], count: int, max_attempts: int,
) -> list[AnnotatedSequence]:
    """Select transcript-oriented samples containing a local TSS."""
    sequences = []
    for sequence in islice(sampler, max_attempts):
        if sequence.feature("tss", required=False) is not None:
            sequences.append(sequence)
            if len(sequences) == count:
                return sequences
    raise ValueError(
        f"Found {len(sequences)} of {count} requested TSS-containing sequences "
        f"within {max_attempts} attempts. Increase --max-attempts or reduce --tss-window."
    )


def _as_test_pair(sequence: AnnotatedSequence) -> SequencePair:
    """Build a reproducible allele fixture while keeping its TSS feature."""
    tss = sequence.resolve_center("tss")
    position = tss + 1 if tss + 1 < len(sequence) else tss - 1
    base = sequence.sequence[position]
    replacement = "C" if base == "A" else "A"
    return SequencePair(
        sequence,
        sequence.replace(position, position + 1, replacement),
        metadata={"local_position": position, "ref": base, "alt": replacement},
    )


def _sample_conditions(directory: Path, count: int, seed: int) -> list[Condition]:
    files = sorted(directory.glob("*.json"))
    if len(files) < count:
        raise ValueError(f"{directory} has {len(files)} JSON files, but {count} descriptions were requested")
    conditions = []
    for path in random.Random(seed).sample(files, count):
        description = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(description, (str, dict)):
            raise ValueError(f"Expected a string or object in {path}")
        conditions.append(Condition(path.name, description))
    return conditions


def _load_runs(path: Path | None, pair_check: bool) -> dict[str, _Run]:
    if path is None:
        return dict(_DEFAULT_PAIR_RUNS if pair_check else _DEFAULT_RUNS)
    specifications = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(specifications, dict) or not specifications or "serial" in specifications:
        raise ValueError("Run configuration must be a nonempty object with 'serial' reserved")
    if any(not name or not isinstance(options, dict) for name, options in specifications.items()):
        raise ValueError("Each run must have a name and an options object")
    return {name: _Run(**options) for name, options in specifications.items()}


def _measurements(prediction: Any, track_channel: int | None) -> Iterator[tuple]:
    value = prediction.scalar("expression")
    if isinstance(value, (list, tuple)):
        raise ValueError("Expected one expression value per prediction")
    yield ("expression", -1, -1, -1, float(value))
    if track_channel is not None:
        shape = prediction.logits.shape
        channel_count = shape[-1] if len(shape) == 3 else 1
        if track_channel >= channel_count:
            raise ValueError(f"Track channel {track_channel} out of range for {channel_count} channels")
        track = prediction.track(channel=track_channel)
        for token, value in zip(track.tokens, track.values, strict=True):
            yield ("track", int(token["input_position"]), int(token["start"]), int(token["end"]), float(value))


def _rows_for_run(
    model: Any,
    sequences: Sequence[AnnotatedSequence | SequencePair],
    conditions: Sequence[Condition],
    run_id: str,
    run: _Run,
    records_per_call: int,
    track_channel: int | None,
    pair_check: bool,
    *,
    show_progress: bool,
) -> Iterator[tuple]:
    n_conditions = len(conditions)
    row_order = range(len(sequences) * n_conditions)
    if run.reverse_order:
        row_order = row_order[::-1]
    retention = RetentionPolicy(
        prediction=PredictionRetention(
            sequence=False,
            logits=track_channel is not None,
            outputs="scalars",
            tokens=track_channel is not None,
            description_tokens=False,
            provenance=False,
        ),
    )
    chunk_count = (len(row_order) + records_per_call - 1) // records_per_call
    for chunk_index, offset in enumerate(range(0, len(row_order), records_per_call), start=1):
        row_ids = row_order[offset : offset + records_per_call]
        if show_progress and chunk_count > 1:
            print(
                f"Run {run_id!r} processing chunk {chunk_index}/{chunk_count} ({len(row_ids):,} rows)",
                file=sys.stderr,
                flush=True,
            )
        inputs = [sequences[row_id // n_conditions] for row_id in row_ids]
        options = dict(
            conditions=[conditions[row_id % n_conditions] for row_id in row_ids],
            center="tss",
            grouping=run.grouping,
            preprocessing_workers=run.preprocessing_workers,
            preprocessing_backend=run.preprocessing_backend,
            prefetch_batches=run.prefetch_batches,
            show_progress=show_progress,
            retention=retention,
        )
        if pair_check:
            batch_limit = "max_pairs_per_forward" if run.pair_execution == "joint" else "max_records_per_forward"
            options[batch_limit] = run.batch_size
            predictions = model.predict_multiple_pairs(inputs, pair_execution=run.pair_execution, **options)
        else:
            predictions = model.predict_multiple_sequences(
                inputs, max_records_per_forward=run.batch_size, return_tokens=track_channel is not None, **options
            )
        if len(predictions) != len(row_ids):
            raise RuntimeError(f"Run {run_id!r} returned {len(predictions)} predictions for {len(row_ids)} rows")
        for row_id, prediction in zip(row_ids, predictions, strict=True):
            outputs = (("ref_", prediction.ref), ("alt_", prediction.alt)) if pair_check else (("", prediction),)
            for prefix, output in outputs:
                for metric, token_pos, start, end, value in _measurements(output, track_channel):
                    finite = math.isfinite(value)
                    yield (
                        row_id, prefix + metric, token_pos, run_id, start, end,
                        value if finite else None, "" if finite else str(value),
                    )


def _measure_all(
    database: sqlite3.Connection,
    sequences: Sequence[AnnotatedSequence | SequencePair],
    conditions: Sequence[Condition],
    runs: Mapping[str, _Run],
    model_options: Mapping[str, Any],
    records_per_call: int,
    track_channel: int | None,
    pair_check: bool,
    *,
    show_progress: bool,
) -> None:
    database.execute(
        """CREATE TABLE observations (
            row_id INTEGER, metric TEXT, token_position INTEGER, run TEXT,
            start INTEGER, end INTEGER, prediction REAL, non_finite TEXT,
            PRIMARY KEY (row_id, metric, token_position, run)
        ) WITHOUT ROWID"""
    )
    model = None
    previous_options = None
    with ExitStack() as model_lifetime:
        for index, (name, run) in enumerate(runs.items(), start=1):
            if show_progress:
                print(
                    f"Starting run {index}/{len(runs)} {name!r} "
                    f"(batch size {run.batch_size}, grouping {run.grouping})",
                    file=sys.stderr,
                    flush=True,
                )
            options = {**model_options, **run.model_overrides}
            if model is None or options != previous_options:
                model_lifetime.close()
                model = None
                model = SequenceModel.load(**options)
                close_model = getattr(model, "close", None)
                if close_model is not None:
                    model_lifetime.callback(close_model)
                previous_options = options
            rows = _rows_for_run(
                model, sequences, conditions, name, run, records_per_call, track_channel, pair_check,
                show_progress=show_progress,
            )
            database.executemany("INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            database.commit()
            if show_progress:
                print(f"Completed run {index}/{len(runs)} {name!r}", file=sys.stderr, flush=True)

    mismatch = database.execute(
        """SELECT row_id, metric, token_position FROM observations
            GROUP BY row_id, metric, token_position
            HAVING count(*) != ? OR min(start) != max(start) OR min(end) != max(end)
            LIMIT 1""",
        (len(runs),),
    ).fetchone()
    if mismatch is not None:
        raise ValueError(f"Output coordinates differ between runs at row, metric, token: {mismatch}")


_COMPARISON_SQL = """
    WITH means AS (
        SELECT row_id, metric, token_position,
            avg(prediction) AS mean_prediction, count(prediction) AS finite_runs,
            max(CASE WHEN run = 'serial' THEN prediction END) AS serial_prediction
        FROM observations GROUP BY row_id, metric, token_position
    )
    SELECT o.run, o.row_id, o.metric, o.token_position, o.start, o.end,
        o.prediction, o.non_finite, m.mean_prediction, m.finite_runs,
        abs(o.prediction - m.mean_prediction) AS absolute_error,
        abs(o.prediction - m.mean_prediction) / max(abs(m.mean_prediction), ?) AS relative_error,
        abs(o.prediction - m.serial_prediction) AS serial_absolute_error
    FROM observations AS o JOIN means AS m USING (row_id, metric, token_position)
"""
_MEASURES = ("prediction", "absolute_error", "relative_error", "serial_absolute_error")


def _summarize(
    database: sqlite3.Connection,
    runs: Mapping[str, _Run],
    epsilon: float,
    output_dir: Path,
) -> dict[str, Any]:
    import numpy as np

    database.execute("CREATE TABLE comparisons AS " + _COMPARISON_SQL, (epsilon,))
    database.execute("CREATE INDEX run_metric ON comparisons(run, metric)")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        cursor = database.execute("SELECT * FROM comparisons ORDER BY row_id, metric, token_position, run")
        writer = csv.writer(stream)
        writer.writerow(column[0] for column in cursor.description)
        writer.writerows(cursor)

    summary: dict[str, Any] = {run: {} for run in runs}
    groups = database.execute(
        "SELECT run, metric, count(*), count(prediction) FROM comparisons GROUP BY run, metric"
    )
    for run, metric, total, finite in groups:
        metrics = {"count": total, "non_finite": total - finite}
        for measure in _MEASURES:
            values = np.fromiter(
                (row[0] for row in database.execute(
                    f"SELECT {measure} FROM comparisons WHERE run = ? AND metric = ? AND {measure} IS NOT NULL",
                    (run, metric),
                )),
                dtype=float,
            )
            quantiles = np.quantile(values, (0, .5, .95, .99, 1), overwrite_input=True) if values.size else ()
            metrics[measure] = dict(zip(("min", "p50", "p95", "p99", "max"), map(float, quantiles)))
            del values
        worst = database.execute(
            """SELECT row_id, token_position, relative_error FROM comparisons
                WHERE run = ? AND metric = ? AND relative_error IS NOT NULL
                ORDER BY relative_error DESC, row_id, token_position LIMIT 1""",
            (run, metric),
        ).fetchone()
        metrics["worst_relative_error"] = (
            dict(zip(("row_id", "token_position", "value"), worst)) if worst else None
        )
        summary[run][metric] = metrics
    return summary


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Each run in --runs can set grouping, batch_size, preprocessing_workers, "
            "preprocessing_backend, prefetch_batches, reverse_order, and model_overrides. "
            "Pair checks also accept pair_execution. "
            "A serial run is always included."
        ),
    )
    parser.add_argument("--model-options", type=Path, required=True, help="JSON keyword arguments for SequenceModel.load")
    parser.add_argument("--runs", type=Path, help="JSON object mapping comparison names to settings")
    parser.add_argument("--descriptions", type=Path, required=True, help="Directory of local JSON descriptions")
    parser.add_argument("--genome", type=Path, required=True, help="Indexed reference FASTA")
    parser.add_argument("--annotation", type=Path, required=True, help="Transcript GTF annotation")
    parser.add_argument("--n-descriptions", type=_positive_int, required=True)
    parser.add_argument("--n-sequences", type=_positive_int, required=True)
    parser.add_argument("--sequence-length", type=_positive_int, required=True)
    parser.add_argument("--tss-window", type=_positive_int, help="Total sampling window in bp, defaults to sequence length")
    parser.add_argument("--max-attempts", type=_positive_int, help="Maximum promoter draws, defaults to 100 per requested sample")
    parser.add_argument("--records-per-call", type=_positive_int, default=256, help="Maximum rows per API call")
    parser.add_argument("--track-channel", type=int, help="Also compare one prediction track channel")
    parser.add_argument("--pair-check", action="store_true", help="Compare pair execution on deterministic REF/ALT fixtures")
    parser.add_argument("--progress", action="store_true", help="Show run status and inference progress bars on stderr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Floor for the relative-error denominator")
    parser.add_argument("--output", type=Path, required=True, help="Directory for manifest.json, predictions.csv, and summary.json")
    args = parser.parse_args(argv)
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("--epsilon must be positive and finite")
    if args.track_channel is not None and args.track_channel < 0:
        parser.error("--track-channel must be nonnegative")
    if args.pair_check and args.sequence_length < 2:
        parser.error("--pair-check requires --sequence-length of at least 2")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Sample the fixed workload and write predictions and numerical diagnostics."""
    args = _parse_args(argv)
    model_options = json.loads(args.model_options.read_text(encoding="utf-8"))
    if not isinstance(model_options, dict):
        raise ValueError("Model options must be a JSON object")
    runs = {"serial": _SERIAL, **_load_runs(args.runs, args.pair_check)}
    if any(run.batch_size > args.records_per_call for run in runs.values()):
        raise ValueError("--records-per-call must be at least as large as every run's batch_size")
    for name, run in runs.items():
        if run.pair_execution == "joint" and (not args.pair_check or run.grouping == "sequence"):
            raise ValueError(f"Run {name!r} requires --pair-check and a joint-compatible grouping")

    output_files = ("manifest.json", "predictions.csv", "summary.json")
    args.output.mkdir(parents=True, exist_ok=True)
    existing = [name for name in output_files if (args.output / name).exists()]
    if existing:
        raise FileExistsError(f"Stress output already exists in {args.output}: {', '.join(existing)}")

    conditions = _sample_conditions(args.descriptions, args.n_descriptions, args.seed)
    with closing(Genome(args.genome, annotation=args.annotation)) as genome:
        sampler = PromoterSampler(
            genome,
            args.sequence_length,
            args.tss_window or args.sequence_length,
            seed=args.seed,
            include_features=False,
        )
        sequences = _take_promoters(sampler, args.n_sequences, args.max_attempts or 100 * args.n_sequences)
    pairs = [_as_test_pair(sequence) for sequence in sequences] if args.pair_check else None
    manifest = {
        "seed": args.seed,
        "genome": str(args.genome.resolve()),
        "annotation": str(args.annotation.resolve()),
        "descriptions": str(args.descriptions.resolve()),
        "sequence_length": args.sequence_length,
        "tss_window": args.tss_window or args.sequence_length,
        "records_per_call": args.records_per_call,
        "epsilon": args.epsilon,
        "track_channel": args.track_channel,
        "pair_check": args.pair_check,
        "row_id": "sequence index * condition count + condition index",
        "model_options": model_options,
        "runs": {name: asdict(run) for name, run in runs.items()},
        "conditions": [condition.to_dict() for condition in conditions],
        "sequences": [
            {
                "name": sequence.name,
                "metadata": sequence.metadata,
                "local_tss": sequence.resolve_center("tss"),
                "sha256": hashlib.sha256(sequence.sequence.encode("ascii")).hexdigest(),
            }
            for sequence in sequences
        ],
    }
    if pairs is not None:
        manifest["pair_edits"] = [pair.metadata for pair in pairs]

    with TemporaryDirectory(prefix=".stress-", dir=args.output) as temporary:
        work = Path(temporary)
        with closing(sqlite3.connect(work / "results.sqlite")) as database:
            _measure_all(
                database, pairs if pairs is not None else sequences, conditions, runs, model_options,
                args.records_per_call, args.track_channel, args.pair_check, show_progress=args.progress,
            )
            summary = _summarize(database, runs, args.epsilon, work)
        for filename, content in (("manifest.json", manifest), ("summary.json", summary)):
            (work / filename).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in output_files:
            (work / name).replace(args.output / name)


if __name__ == "__main__":
    main()
