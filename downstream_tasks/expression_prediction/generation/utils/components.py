"""Factories for gena_expression models and optimization components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from gena_expression import (
    ExpressionScorer,
    LinearCombinationScorer,
    MinOnTargetMaxOffTargetFitness,
    ParallelCrossoverMutationGenerator,
    SequenceModel,
    SequentialCrossoverMutationGenerator,
    TrackFeatureScorer,
    TrackWindowScorer,
)

from .config import ConfigError, resource_value


def build_model(config: Mapping[str, Any]) -> SequenceModel:
    """Load the configured model and its native Hydra configuration."""

    if config["genalm_home"] is not None:
        os.environ["GENALM_HOME"] = str(config["genalm_home"])
    return SequenceModel.load(
        model_cls=config["model_cls"],
        checkpoint=config["checkpoint"],
        config=config["config"],
        dna_tokenizer=resource_value(config["dna_tokenizer"], label="model.dna_tokenizer"),
        description_tokenizer=resource_value(
            config["description_tokenizer"],
            label="model.description_tokenizer",
        ),
        dna_max_seq_len=config["dna_max_seq_len"],
        desc_max_seq_len=config["desc_max_seq_len"],
        token_len_for_fetch=config["token_len_for_fetch"],
        num_before=config["num_before"],
        device=config["device"],
        output_names=config["output_names"],
        description_formatter=config["description_formatter"],
    )


def build_scorer(config: Mapping[str, Any]) -> Any:
    """Construct one explicitly supported scalar scorer."""

    scorer_type = config["type"]
    params = dict(config["params"])
    if scorer_type == "expression":
        return ExpressionScorer(**params)
    if scorer_type == "track_feature":
        return TrackFeatureScorer(**params)
    if scorer_type == "track_window":
        return TrackWindowScorer(**params)
    if scorer_type == "linear_combination":
        name = params.pop("name", "linear_combination")
        component_specs = params.pop("components", None)
        if params or not isinstance(component_specs, Mapping) or not component_specs:
            raise ConfigError(
                "linear_combination scorer params must contain only name and non-empty components."
            )
        components = {}
        for component_name, component in component_specs.items():
            if not isinstance(component, Mapping) or "scorer" not in component or "coefficient" not in component:
                raise ConfigError(
                    f"scorer.params.components.{component_name} requires scorer and coefficient."
                )
            components[str(component_name)] = (
                build_scorer(component["scorer"]),
                component["coefficient"],
            )
        return LinearCombinationScorer(components, name=name)
    raise ConfigError(f"Unsupported scorer.type: {scorer_type!r}.")


def build_fitness(config: Mapping[str, Any]) -> Any:
    """Construct the configured target-score fitness function."""

    fitness_type = config["type"]
    params = dict(config["params"])
    if fitness_type == "mean_on_minus_mean_off":
        if params:
            raise ConfigError("mean_on_minus_mean_off does not accept fitness.params.")
        return None
    if fitness_type == "min_on_max_off":
        return MinOnTargetMaxOffTargetFitness(**params)
    raise ConfigError(f"Unsupported fitness.type: {fitness_type!r}.")


def build_offspring_generator(config: Mapping[str, Any]) -> Any:
    """Construct the configured sequential or parallel offspring generator."""

    generator_type = config["type"]
    params = dict(config["params"])
    if generator_type == "sequential":
        return SequentialCrossoverMutationGenerator(**params)
    if generator_type == "parallel":
        return ParallelCrossoverMutationGenerator(**params)
    raise ConfigError(f"Unsupported offspring.type: {generator_type!r}.")
