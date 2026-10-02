"""Static consistency checks for every configs/cnn_*.yaml.

Catches the mistakes that otherwise only surface minutes into a cluster run:
a mistyped dataset kwarg, a model kwarg the class does not accept, or a window
length that disagrees with the CNN stride implied by the model configuration.

Signatures are read with :mod:`ast` instead of by importing, so this runs
without pysam, pyBigWig, h5py or a downloaded checkpoint.
"""

import ast
import os
from pathlib import Path

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"

DATASET_TARGET = "downstream_tasks.expression_prediction.expression_dataset_cnn.ExpressionDatasetCNN"
MODEL_TARGET = "downstream_tasks.expression_prediction.expression_model_cnn:ExpressionCountsCNN"

ALLOWED_DATASET_DESCRIPTIONS = {
    "Expression_dataset_v1_GRCh38_csv dataset",
    "Expression_dataset_v1_mm10_CPM dataset",
}


def _init_parameters(source_path: Path, class_name: str):
    """Names of ``__init__`` keyword parameters of a class, without importing it."""
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args
                    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
                    return set(names) - {"self"}
    raise AssertionError(f"{class_name}.__init__ not found in {source_path}")


def load_config(path: Path):
    os.environ.setdefault("GENALM_HOME", "/tmp/genalm_home")
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return cfg


def all_configs():
    paths = sorted(CONFIG_DIR.glob("cnn_*.yaml"))
    assert paths, f"no cnn_*.yaml under {CONFIG_DIR}"
    return [(p.name, load_config(p)) for p in paths]


def _dataset_blocks(cfg):
    return {
        key: value
        for key, value in cfg.items()
        if str(key).startswith(("train_dataset", "valid_dataset"))
    }


def test_configs_parse_and_resolve():
    for name, cfg in all_configs():
        assert cfg.args_params.model_cls == MODEL_TARGET, name
        assert len(_dataset_blocks(cfg)) == 4, (name, list(_dataset_blocks(cfg)))


def test_dataset_kwargs_match_signature():
    allowed = _init_parameters(HERE / "expression_dataset_cnn.py", "ExpressionDatasetCNN")
    for name, cfg in all_configs():
        shared = set(cfg.shared_dataset_params.keys())
        for block_name, block in _dataset_blocks(cfg).items():
            assert block["_target_"] == DATASET_TARGET, (name, block_name)
            keys = (set(block.keys()) | shared) - {"_target_", "repeat_factor"}
            unknown = keys - allowed
            assert not unknown, (
                f"{name}:{block_name} passes unknown dataset kwargs: {sorted(unknown)}"
            )


def test_model_kwargs_match_signature():
    allowed = _init_parameters(HERE / "expression_model_cnn.py", "ExpressionCountsCNN")
    for name, cfg in all_configs():
        unknown = (set(cfg.model_kwargs.keys()) - {"_target_"}) - allowed
        assert not unknown, f"{name} passes unknown model kwargs: {sorted(unknown)}"


def test_required_model_kwargs_present():
    for name, cfg in all_configs():
        for required in ("config", "hf_model_name_decoder"):
            assert required in cfg.model_kwargs, f"{name} is missing model_kwargs.{required}"


def test_window_stride_and_seq_len_agree():
    """input_seq_len must equal CLS + bins + SEP, and the stride must match the model."""
    for name, cfg in all_configs():
        mk, shared = cfg.model_kwargs, cfg.shared_dataset_params
        implied = int(mk.cnn_pool_stride) ** int(mk.cnn_num_pooling_stages)
        assert int(shared.cnn_total_stride) == implied, (
            f"{name}: cnn_total_stride={shared.cnn_total_stride} but the model implies {implied}"
        )
        window = int(shared.dna_window_len)
        assert window % implied == 0, f"{name}: dna_window_len={window} not divisible by {implied}"
        expected = window // implied + 2
        assert int(cfg.args_params.input_seq_len) == expected, (
            f"{name}: input_seq_len={cfg.args_params.input_seq_len} but CLS + "
            f"{window // implied} bins + SEP = {expected}"
        )


def test_n_keys_is_shared_across_datasets():
    """The collate stacks a (B, n_keys, ...) axis, so one common value is required."""
    for name, cfg in all_configs():
        shared = cfg.shared_dataset_params
        assert "n_keys" in shared, (
            f"{name}: shared_dataset_params must set n_keys -- a ConcatDataset of human "
            f"and mouse would otherwise yield items the collate cannot stack"
        )
        n_keys = int(shared.n_keys)
        assert n_keys >= 1, (name, n_keys)
        for block_name, block in _dataset_blocks(cfg).items():
            if "n_keys" in block:
                assert int(block["n_keys"]) == n_keys, (
                    f"{name}:{block_name} overrides n_keys={block['n_keys']} against {n_keys}"
                )


def test_optimize_metric_is_produced_by_the_runner():
    """optimize_metric must name a dataset the metric code special-cases."""
    for name, cfg in all_configs():
        metric = cfg.args_params.optimize_metric
        assert metric.startswith("score_predictions_"), (name, metric)
        desc = metric[len("score_predictions_"):]
        assert desc in ALLOWED_DATASET_DESCRIPTIONS, (
            f"{name}: {desc!r} is not in the runner's ALLOWED set, so score_predictions_* "
            f"would never be logged and save_best would never fire"
        )


def test_dropout_config_is_an_isolated_ab():
    """cnn_v1_dp must differ from cnn_v1 only in the regulariser and the budget.

    The point of the experiment is to attribute any change to dropout, so a
    second accidental difference would invalidate it.
    """
    cfgs = dict(all_configs())
    if "cnn_v1.yaml" not in cfgs or "cnn_v1_dp.yaml" not in cfgs:
        return
    base, dp = cfgs["cnn_v1.yaml"], cfgs["cnn_v1_dp.yaml"]

    assert float(dp.model_kwargs.cnn_dropout) > 0, "cnn_v1_dp must actually enable dropout"
    assert float(base.model_kwargs.get("cnn_dropout", 0.0)) == 0.0, "baseline must have none"

    intended = {"cnn_dropout", "cnn_dropout_channels"}
    differing = {
        k for k in set(base.model_kwargs) | set(dp.model_kwargs)
        if base.model_kwargs.get(k) != dp.model_kwargs.get(k)
    }
    assert differing <= intended, (
        f"model_kwargs differ beyond dropout: {sorted(differing - intended)}"
    )

    # The training budget may differ; the optimisation recipe may not.
    for key in ("lr", "seed", "input_seq_len", "optimizer", "weight_decay",
                "lr_scheduler", "num_warmup_steps", "optimize_metric"):
        assert base.args_params[key] == dp.args_params[key], f"args_params.{key} differs"

    assert base.TASK_NAME != dp.TASK_NAME, "the two runs would share an output directory"


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
