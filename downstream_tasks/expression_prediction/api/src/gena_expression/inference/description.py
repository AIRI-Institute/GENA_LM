"""Description-formatter discovery for configured GENA-LM datasets."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._import_utils import temporary_sys_path


def get_make_description_from_json(
    config_path: str | Path,
) -> Callable[[dict[str, Any], str | None, str | None], str]:
    """Return the configured dataset class's description function.

    ``get_class`` imports the class named by ``*target*`` but, unlike
    ``hydra.utils.instantiate``, does not call its constructor. Imports stay
    local so importing :mod:`gena_expression` does not require model extras.
    """

    from hydra.utils import get_class
    from omegaconf import DictConfig, OmegaConf

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


def get_configured_description_formatter(
    config_path: str | Path,
) -> Callable[..., str]:
    """Load the configured dataset formatter with GENA-LM import visibility."""

    genalm_home = os.environ.get("GENALM_HOME")
    if not genalm_home:
        raise EnvironmentError(
            "GENALM_HOME is required to obtain the description formatter "
            "from the configured dataset class."
        )

    genalm_root = (Path(genalm_home).expanduser() / "GENA_LM").resolve()
    if not genalm_root.is_dir():
        raise FileNotFoundError(f"GENA_LM directory does not exist: {genalm_root}")

    with temporary_sys_path(genalm_root):
        return get_make_description_from_json(config_path)
