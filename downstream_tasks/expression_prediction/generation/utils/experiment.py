"""High-level orchestration kept out of the command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gena_expression import SequenceOptimizer

from .components import build_fitness, build_model, build_offspring_generator, build_scorer
from .config import RunConfig
from .inputs import (
    build_genome,
    build_plasmid_template,
    build_promoter_sampler,
    collect_promoter_population,
    load_targets,
)
from .results import write_results


def prepare_output_directory(config: dict[str, Any]) -> Path:
    """Create an output directory without silently mixing independent runs."""

    output_dir = Path(config["directory"])
    if output_dir.exists() and any(output_dir.iterdir()) and not config["allow_existing"]:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Set output.allow_existing=true "
            "only when overwriting the known result files is intentional."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_optimizer(
    config: RunConfig,
    *,
    template: Any,
    sampled_population: tuple[Any, ...],
    on_targets: dict[str, Any],
    off_targets: dict[str, Any],
    scorer: Any,
    fitness: Any,
    offspring_generator: Any,
) -> SequenceOptimizer:
    """Construct the optimizer with one sampled promoter already in the plasmid."""

    optimization = config.section("optimization")
    plasmid = config.section("plasmid")
    return SequenceOptimizer(
        sequence=template,
        region=plasmid["region_feature_name"],
        tokenization_center=plasmid["tokenization_center_name"],
        scorer=scorer,
        on_targets=on_targets,
        off_targets=off_targets,
        offspring_generator=offspring_generator,
        offspring_generator_mode=optimization["offspring_generator_mode"],
        fitness_function=fitness,
        population_size=optimization["population_size"],
        offspring_size=optimization["offspring_size"],
        n_generations=optimization["n_generations"],
        initial_population=[sample.sequence for sample in sampled_population[1:]],
        selection_strategy=optimization["selection_strategy"],
        elite_count=optimization["elite_count"],
        tournament_size=optimization["tournament_size"],
        random_seed=optimization["random_seed"],
    )


def run_experiment(config: RunConfig) -> list[Path]:
    """Sample promoters from the genome, paste one into the plasmid, and evolve."""

    output_dir = prepare_output_directory(config.section("output"))
    on_targets, off_targets = load_targets(config.section("conditions"))

    genome = build_genome(config.section("genome"))
    try:
        sampler_config = config.section("sampler")
        sampler = build_promoter_sampler(genome, sampler_config)
        sampled_population = collect_promoter_population(
            sampler,
            population_size=config.section("optimization")["population_size"],
            seq_len=sampler_config["seq_len"],
            unique_sequences=sampler_config["unique_sequences"],
            max_attempts=sampler_config["max_attempts"],
        )
    finally:
        genome.close()

    template = build_plasmid_template(config.section("plasmid"), sampled_population[0])
    scorer = build_scorer(config.section("scorer"))
    fitness = build_fitness(config.section("fitness"))
    offspring_generator = build_offspring_generator(config.section("offspring"))
    try:
        model = build_model(config.section("model"))
        optimizer = build_optimizer(
            config,
            template=template,
            sampled_population=sampled_population,
            on_targets=on_targets,
            off_targets=off_targets,
            scorer=scorer,
            fitness=fitness,
            offspring_generator=offspring_generator,
        )
        runtime = config.section("runtime")
        optimizer.run_evolution(model=model, **runtime)
    finally:
        close = getattr(offspring_generator, "close", None)
        if callable(close):
            close()

    return write_results(
        optimizer,
        output_dir,
        config=config,
        sampled_population=sampled_population,
    )
