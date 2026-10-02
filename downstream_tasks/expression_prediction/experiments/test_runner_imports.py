"""Import-only check for the training entry point.

The unit tests exercise the CNN encoder, the model and the config, but none of
them imports the runner -- it drags in accelerate, the Slurm plumbing and
lm_experiments_tools. That gap cost a 45-second, 8-GPU job on front3, which died
with ModuleNotFoundError: lm_experiments_tools (not installed into the venv; it
is picked up from the t5-experiments checkout via PYTHONPATH).

This test is deliberately cheap and needs no GPU and no data, so it can run on a
login node before submitting anything.
"""

import importlib

MODULES = [
    "downstream_tasks.expression_prediction.alphagenome_cnn",
    "downstream_tasks.expression_prediction.expression_dataset_cnn",
    "downstream_tasks.expression_prediction.expression_model_cnn",
    "downstream_tasks.expression_prediction.run_expression_finetuning_cnn",
]


def test_runner_and_its_dependencies_import():
    for name in MODULES:
        importlib.import_module(name)


def test_trainer_is_reachable():
    """lm_experiments_tools lives outside the venv; PYTHONPATH must reach it."""
    from lm_experiments_tools import TrainerAccelerate, TrainerAccelerateArgs  # noqa: F401


def test_metric_helpers_are_reachable():
    from downstream_tasks.expression_prediction.datasets.src.score_ct_specificity import (  # noqa: F401
        mean_and_residuals_correlation,
        score_predictions,
    )


def test_runner_exposes_what_the_job_needs():
    m = importlib.import_module(
        "downstream_tasks.expression_prediction.run_expression_finetuning_cnn"
    )
    for attr in ("main", "load_selected_gene_ids", "build_loader", "build_dataset_from_cfgs"):
        assert hasattr(m, attr), f"runner is missing {attr}"


if __name__ == "__main__":
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)
