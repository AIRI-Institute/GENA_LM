"""Tests for the fixed-workload inference stress runner."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from gena_expression import AnnotatedSequence
from gena_expression.stress import _take_promoters, main


class _FakeModel:
    def __init__(self, shift=0, *, shift_coordinates=False, truncate=False):
        self.shift = shift
        self.shift_coordinates = shift_coordinates
        self.truncate = truncate
        self.calls = []
        self.closed = False

    def close(self):
        self.closed = True

    def predict_multiple_sequences(self, sequences, *, conditions, **options):
        self.calls.append((tuple(condition.name for condition in conditions), options))
        predictions = []
        for sequence, condition in zip(sequences, conditions, strict=True):
            assert sequence.resolve_center(options["center"]) == 6
            value = sequence.metadata["tss"] + int(condition.description)
            if condition.description == "2" and self.shift:
                value = float("nan")
            else:
                value += self.shift
            token = {"input_position": 1, "start": int(self.shift_coordinates), "end": 2}
            predictions.append(
                SimpleNamespace(
                    logits=SimpleNamespace(shape=(1, 2, 1)),
                    scalar=lambda _, value=value: value,
                    track=lambda *, channel, token=token, value=value: SimpleNamespace(tokens=[token], values=[value / 2]),
                )
            )
        return predictions[:-1] if self.truncate else predictions

    def predict_multiple_pairs(self, pairs, *, conditions, pair_execution, **options):
        assert ("max_pairs_per_forward" in options) == (pair_execution == "joint")
        assert ("max_records_per_forward" in options) == (pair_execution == "separate")
        for pair in pairs:
            assert pair.ref.sequence != pair.alt.sequence
            assert pair.ref.resolve_center("tss") == pair.alt.resolve_center("tss")
        refs = self.predict_multiple_sequences([pair.ref for pair in pairs], conditions=conditions, **options)
        alts = self.predict_multiple_sequences([pair.alt for pair in pairs], conditions=conditions, **options)
        return [SimpleNamespace(ref=ref, alt=alt) for ref, alt in zip(refs, alts, strict=True)]


@pytest.fixture
def workload(tmp_path):
    import pysam

    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "ACGT" * 100 + "\n", encoding="utf-8")
    pysam.faidx(str(fasta))
    annotation = tmp_path / "annotation.gtf"
    annotation.write_text(
        'chr1\tfixture\ttranscript\t31\t50\t.\t+\t.\tgene_id "g1"; transcript_id "t1";\n'
        'chr1\tfixture\ttranscript\t100\t121\t.\t-\t.\tgene_id "g2"; transcript_id "t2";\n',
        encoding="utf-8",
    )
    annotation = pysam.tabix_index(str(annotation), preset="gff")
    # Use an odd length so the TSS is centered on both strands.
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    for value in (0, 2):
        (descriptions / f"{value}.json").write_text(json.dumps(str(value)), encoding="utf-8")
    options = tmp_path / "model.json"
    options.write_text('{"device": "cpu"}', encoding="utf-8")
    runs = tmp_path / "runs.json"
    runs.write_text(
        '{"reversed": {"grouping": "sequence", "reverse_order": true, "batch_size": 2,'
        ' "preprocessing_workers": 2, "preprocessing_backend": "thread", "prefetch_batches": 3,'
        ' "model_overrides": {"precision": "float32"}}}',
        encoding="utf-8",
    )
    output = tmp_path / "output"
    argv = [
        "--model-options", str(options), "--runs", str(runs),
        "--descriptions", str(descriptions), "--genome", str(fasta), "--annotation", str(annotation),
        "--n-descriptions", "2", "--n-sequences", "2", "--sequence-length", "13",
        "--seed", "14", "--output", str(output), "--track-channel", "0",
    ]
    return argv, output


def test_stress_compares_fixed_input_ids_and_reports_drift(monkeypatch, workload):
    argv, output = workload
    models = []

    def load(*, device, precision=None):
        assert device == "cpu"
        model = _FakeModel(shift=2 if precision == "float32" else 0)
        models.append(model)
        return model

    monkeypatch.setattr("gena_expression.stress.SequenceModel.load", load)
    main(argv)

    manifest = json.loads((output / "manifest.json").read_text())
    assert [sequence["metadata"]["strand"] for sequence in manifest["sequences"]] == ["+", "-"]
    assert [entry["name"] for entry in manifest["conditions"]] == list(models[0].calls[0][0][:2])
    assert models[0].calls[0][0] == models[1].calls[0][0][::-1]
    assert models[1].calls[0][1]["preprocessing_workers"] == 2
    assert models[1].calls[0][1]["prefetch_batches"] == 3
    assert all(model.closed for model in models)

    with (output / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    condition_index = [entry["description"] for entry in manifest["conditions"]].index("0")
    row = next(
        row for row in rows
        if row["row_id"] == str(condition_index) and row["run"] == "reversed" and row["metric"] == "expression"
    )
    assert float(row["prediction"]) == 32.0
    assert float(row["mean_prediction"]) == 31.0
    assert float(row["relative_error"]) == pytest.approx(1 / 31)
    assert float(row["serial_absolute_error"]) == 2.0

    summary = json.loads((output / "summary.json").read_text())
    expression = summary["reversed"]["expression"]
    assert expression["count"] == 4
    assert expression["non_finite"] == 2
    assert expression["worst_relative_error"]["row_id"] == condition_index
    assert expression["serial_absolute_error"]["p95"] == 2.0
    assert summary["reversed"]["track"]["count"] == 4
    assert summary["reversed"]["track"]["non_finite"] == 2


@pytest.mark.parametrize("failure", ["coordinates", "count"])
def test_failed_run_cleans_up_without_publishing_reports(monkeypatch, workload, failure):
    argv, output = workload
    monkeypatch.setattr(
        "gena_expression.stress.SequenceModel.load",
        lambda *, device, precision=None: _FakeModel(
            shift_coordinates=bool(precision) and failure == "coordinates",
            truncate=bool(precision) and failure == "count",
        ),
    )
    if failure == "coordinates":
        with pytest.raises(ValueError, match="Output coordinates differ"):
            main(argv)
    else:
        with pytest.raises(RuntimeError, match="returned 3 predictions for 4 rows"):
            main(argv)
    assert list(output.iterdir()) == []


def test_promoter_sampling_is_bounded_when_local_tss_is_missing():
    sequence = AnnotatedSequence("ACGT")
    with pytest.raises(ValueError, match="Found 0 of 1 requested.*within 3 attempts"):
        _take_promoters([sequence] * 5, 1, 3)


def test_pair_checks_compare_reference_and_alternative_outputs(monkeypatch, workload):
    argv, output = workload
    runs_option = argv.index("--runs")
    argv = argv[:runs_option] + argv[runs_option + 2:] + ["--pair-check"]
    monkeypatch.setattr("gena_expression.stress.SequenceModel.load", lambda **options: _FakeModel())
    main(argv)
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary) == {"serial", "separate", "joint"}
    assert set(summary["joint"]) == {"ref_expression", "alt_expression", "ref_track", "alt_track"}
    assert summary["joint"]["ref_expression"]["serial_absolute_error"]["max"] == 0
    assert summary["joint"]["alt_expression"]["count"] == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(edit["local_position"] == 7 for edit in manifest["pair_edits"])


def test_track_check_rejects_missing_channel(monkeypatch, workload):
    argv, output = workload
    argv[argv.index("--track-channel") + 1] = "3"
    monkeypatch.setattr("gena_expression.stress.SequenceModel.load", lambda **options: _FakeModel())
    with pytest.raises(ValueError, match="Track channel 3 out of range for 1 channels"):
        main(argv)
    assert list(output.iterdir()) == []
