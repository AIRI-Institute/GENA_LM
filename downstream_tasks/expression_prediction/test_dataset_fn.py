"""Get a dataset helper method from a Hydra config without creating a dataset.

Example (run from the ``GENA_LM`` directory)::

    python -m downstream_tasks.expression_prediction.test_dataset_fn
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "final_2048.yaml"


def get_make_description_from_json(
    config_path: str | Path,
) -> Callable[[dict[str, Any], str, str], str]:
    """Return the configured dataset class's description function.

    ``get_class`` imports the class named by ``_target_`` but, unlike
    ``hydra.utils.instantiate``, does not call its constructor.
    """
    config = OmegaConf.load(config_path)

    dataset_config = next(
        (
            key
            for prefix in ("train_dataset_", "valid_dataset_")
            for key in config.keys()
            if key.startswith(prefix)
        ),
        None,
    )
    if dataset_config is None:
        raise KeyError(
            f"No train_dataset_* or valid_dataset_* config was found in {config_path}"
        )

    dataset: DictConfig = config[dataset_config]

    target = OmegaConf.select(dataset, "_target_")
    if not isinstance(target, str) or not target:
        raise ValueError(f"Dataset config {dataset_config!r} has no valid _target_")

    dataset_class = get_class(target)
    make_description = getattr(dataset_class, "make_description_from_json", None)
    if not callable(make_description):
        raise AttributeError(f"{target} has no callable make_description_from_json")

    return make_description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Hydra YAML config (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    make_description = get_make_description_from_json(args.config)

    # The returned callable can now be used without an ExpressionDataset object.
    example = make_description(
        {"cell_type": "T_cell", "source": "example"},
        "example-id",
        "example.json",
    )
    print(example)


if __name__ == "__main__":
    main()
