"""Configuration loading, validation, and config-relative path handling."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


_REQUIRED_SECTIONS = (
    "project",
    "model",
    "genome",
    "sampler",
    "plasmid",
    "conditions",
    "scorer",
    "fitness",
    "offspring",
    "optimization",
    "runtime",
    "output",
)

_REQUIRED_KEYS = {
    "project": ("name",),
    "model": (
        "model_cls",
        "checkpoint",
        "config",
        "dna_tokenizer",
        "description_tokenizer",
        "dna_max_seq_len",
        "desc_max_seq_len",
        "token_len_for_fetch",
        "num_before",
        "device",
        "output_names",
        "description_formatter",
        "genalm_home",
    ),
    "genome": ("fasta_path", "annotation_path", "build", "chrom_style", "cache"),
    "sampler": (
        "seq_len",
        "tss_sampling_window",
        "seed",
        "transcript_feature_types",
        "include_features",
        "unique_sequences",
        "max_attempts",
    ),
    "plasmid": (
        "genbank_path",
        "fasta_path",
        "reporter_feature_name",
        "replacement_feature_name",
        "region_feature_name",
        "tokenization_center_name",
        "circular",
        "window_size",
    ),
    "conditions": ("seed", "strict", "search_fields", "on_target", "off_target"),
    "scorer": ("type", "params"),
    "fitness": ("type", "params"),
    "offspring": ("type", "params"),
    "optimization": (
        "population_size",
        "offspring_size",
        "n_generations",
        "selection_strategy",
        "elite_count",
        "tournament_size",
        "offspring_generator_mode",
        "random_seed",
    ),
    "runtime": (
        "evaluation_chunk_size",
        "grouping",
        "preprocessing_workers",
        "preprocessing_backend",
        "max_records_per_forward",
        "prefetch_batches",
        "show_progress",
    ),
    "output": ("directory", "allow_existing"),
}


def _resolve_path(base: Path, value: str | Path) -> Path:
    """Resolve one filesystem path relative to the configuration directory."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return a mutable mapping or raise a configuration error."""

    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a mapping.")
    return dict(value)


def _require_positive_int(value: Any, *, label: str) -> None:
    """Validate one positive integer configuration value."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer.")


def _resolve_model_class(base: Path, value: Any) -> str:
    """Resolve the file portion of ``path.py::Class`` while preserving modules."""

    if not isinstance(value, str) or "::" not in value:
        raise ConfigError("model.model_cls must use 'path-or-module::ClassName'.")
    source, class_name = value.split("::", 1)
    if not source or not class_name:
        raise ConfigError("model.model_cls must include both source and class name.")
    source_path = Path(source)
    if source_path.suffix == ".py" or source_path.is_absolute() or source.startswith("."):
        source = str(_resolve_path(base, source_path))
    return f"{source}::{class_name}"


def _resolve_resource(base: Path, value: Any, *, label: str) -> Any:
    """Resolve ``{path: ...}`` resources and preserve identifiers or nulls."""

    if value is None or isinstance(value, str):
        return value
    resource = _require_mapping(value, label=label)
    choices = [key for key in ("path", "identifier") if key in resource]
    if len(choices) != 1:
        raise ConfigError(f"{label} must contain exactly one of 'path' or 'identifier'.")
    key = choices[0]
    if not isinstance(resource[key], str) or not resource[key]:
        raise ConfigError(f"{label}.{key} must be a non-empty string.")
    if key == "path":
        resource[key] = _resolve_path(base, resource[key])
    return resource


def resource_value(value: Any, *, label: str) -> str | None:
    """Return the path or identifier accepted by ``SequenceModel.load``."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    resource = _require_mapping(value, label=label)
    if "path" in resource:
        return str(resource["path"])
    if "identifier" in resource:
        return str(resource["identifier"])
    raise ConfigError(f"{label} does not contain a usable resource value.")


def json_ready(value: Any) -> Any:
    """Convert paths and nested containers to JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def _validate(data: dict[str, Any]) -> None:
    """Validate the complete, explicit experiment configuration."""

    for section_name in _REQUIRED_SECTIONS:
        if section_name not in data:
            raise ConfigError(f"Missing required section: {section_name}.")
        data[section_name] = _require_mapping(data[section_name], label=section_name)

    for section_name, keys in _REQUIRED_KEYS.items():
        section = data[section_name]
        missing = [key for key in keys if key not in section]
        if missing:
            joined = ", ".join(f"{section_name}.{key}" for key in missing)
            raise ConfigError(f"Missing required configuration value(s): {joined}.")

    for role in ("on_target", "off_target"):
        group = _require_mapping(data["conditions"][role], label=f"conditions.{role}")
        for key in ("directory", "keys", "filters", "key_filters"):
            if key not in group:
                raise ConfigError(f"Missing required configuration value: conditions.{role}.{key}.")
        if not isinstance(group["keys"], list) or not all(
            isinstance(key, str) and key for key in group["keys"]
        ):
            raise ConfigError(f"conditions.{role}.keys must be a list of non-empty strings.")
        for mapping_name in ("filters", "key_filters"):
            if not isinstance(group[mapping_name], Mapping):
                raise ConfigError(f"conditions.{role}.{mapping_name} must be a mapping.")
        data["conditions"][role] = group

    if not data["conditions"]["on_target"]["keys"] and not data["conditions"]["off_target"]["keys"]:
        raise ConfigError("At least one on-target or off-target condition key is required.")
    search_fields = data["conditions"]["search_fields"]
    if search_fields is not None and (
        not isinstance(search_fields, list)
        or not all(isinstance(field, str) and field for field in search_fields)
    ):
        raise ConfigError("conditions.search_fields must be null or a list of non-empty strings.")

    for section_name in ("scorer", "fitness", "offspring"):
        if not isinstance(data[section_name]["type"], str) or not data[section_name]["type"]:
            raise ConfigError(f"{section_name}.type must be a non-empty string.")
        if not isinstance(data[section_name]["params"], Mapping):
            raise ConfigError(f"{section_name}.params must be a mapping.")

    if not isinstance(data["project"]["name"], str) or not data["project"]["name"].strip():
        raise ConfigError("project.name must be a non-empty string.")

    for label, value in (
        ("model.token_len_for_fetch", data["model"]["token_len_for_fetch"]),
        ("model.num_before", data["model"]["num_before"]),
        ("sampler.seq_len", data["sampler"]["seq_len"]),
        ("sampler.tss_sampling_window", data["sampler"]["tss_sampling_window"]),
        ("sampler.max_attempts", data["sampler"]["max_attempts"]),
        ("optimization.population_size", data["optimization"]["population_size"]),
        ("optimization.offspring_size", data["optimization"]["offspring_size"]),
        ("optimization.tournament_size", data["optimization"]["tournament_size"]),
        ("runtime.evaluation_chunk_size", data["runtime"]["evaluation_chunk_size"]),
    ):
        _require_positive_int(value, label=label)

    for label in ("dna_max_seq_len", "desc_max_seq_len"):
        value = data["model"][label]
        if value is not None:
            _require_positive_int(value, label=f"model.{label}")

    if data["sampler"]["tss_sampling_window"] < data["sampler"]["seq_len"]:
        raise ConfigError("sampler.tss_sampling_window must be at least sampler.seq_len.")
    if data["sampler"]["max_attempts"] < data["optimization"]["population_size"]:
        raise ConfigError("sampler.max_attempts must be at least optimization.population_size.")

    generations = data["optimization"]["n_generations"]
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 0:
        raise ConfigError("optimization.n_generations must be a non-negative integer.")
    workers = data["runtime"]["preprocessing_workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ConfigError("runtime.preprocessing_workers must be a non-negative integer.")
    prefetch = data["runtime"]["prefetch_batches"]
    if isinstance(prefetch, bool) or not isinstance(prefetch, int) or prefetch < 0:
        raise ConfigError("runtime.prefetch_batches must be a non-negative integer.")
    max_records = data["runtime"]["max_records_per_forward"]
    if max_records is not None:
        _require_positive_int(max_records, label="runtime.max_records_per_forward")

    window_size = data["plasmid"]["window_size"]
    if window_size != "full":
        _require_positive_int(window_size, label="plasmid.window_size")

    elite_count = data["optimization"]["elite_count"]
    population_size = data["optimization"]["population_size"]
    if (
        isinstance(elite_count, bool)
        or not isinstance(elite_count, int)
        or not 0 <= elite_count <= population_size
    ):
        raise ConfigError(
            "optimization.elite_count must be between zero and optimization.population_size."
        )

    if data["optimization"]["selection_strategy"] not in {
        "top_k",
        "tournament",
        "rank_weighted",
    }:
        raise ConfigError("optimization.selection_strategy is not supported.")
    if data["optimization"]["offspring_generator_mode"] not in {"sequences", "records"}:
        raise ConfigError("optimization.offspring_generator_mode must be sequences or records.")
    if data["runtime"]["grouping"] not in {
        "serial",
        "no_grouping",
        "condition",
        "sequence",
        "auto",
    }:
        raise ConfigError("runtime.grouping is not supported.")
    if data["runtime"]["preprocessing_backend"] not in {"process", "thread"}:
        raise ConfigError("runtime.preprocessing_backend must be process or thread.")

    for section_name, key in (
        ("genome", "cache"),
        ("sampler", "include_features"),
        ("sampler", "unique_sequences"),
        ("plasmid", "circular"),
        ("conditions", "strict"),
        ("runtime", "show_progress"),
        ("output", "allow_existing"),
    ):
        if not isinstance(data[section_name][key], bool):
            raise ConfigError(f"{section_name}.{key} must be a boolean.")


def _resolve_paths(data: dict[str, Any], base: Path) -> None:
    """Resolve every declared local path in place."""

    model = data["model"]
    model["model_cls"] = _resolve_model_class(base, model["model_cls"])
    for key in ("checkpoint", "config"):
        model[key] = _resolve_path(base, model[key])
    if model["genalm_home"] is not None:
        model["genalm_home"] = _resolve_path(base, model["genalm_home"])
    for key in ("dna_tokenizer", "description_tokenizer"):
        model[key] = _resolve_resource(base, model[key], label=f"model.{key}")

    for key in ("fasta_path", "annotation_path"):
        data["genome"][key] = _resolve_path(base, data["genome"][key])
    for key in ("genbank_path", "fasta_path"):
        value = data["plasmid"][key]
        data["plasmid"][key] = None if value is None else _resolve_path(base, value)
    for role in ("on_target", "off_target"):
        group = data["conditions"][role]
        group["directory"] = _resolve_path(base, group["directory"])
    data["output"]["directory"] = _resolve_path(base, data["output"]["directory"])


@dataclass(frozen=True)
class RunConfig:
    """Validated experiment configuration with config-relative paths resolved."""

    source_path: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        """Load, validate, and resolve one YAML configuration."""

        source_path = Path(path).expanduser().resolve(strict=True)
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        data = _require_mapping(loaded, label="configuration root")
        _validate(data)
        _resolve_paths(data, source_path.parent)
        return cls(source_path=source_path, data=data)

    def section(self, name: str) -> dict[str, Any]:
        """Return a defensive copy of one named configuration section."""

        if name not in self.data:
            raise ConfigError(f"Unknown configuration section: {name}.")
        return copy.deepcopy(self.data[name])

    def as_dict(self) -> dict[str, Any]:
        """Return the resolved configuration in JSON-serializable form."""

        return {
            "config_source": str(self.source_path),
            **json_ready(copy.deepcopy(self.data)),
        }
