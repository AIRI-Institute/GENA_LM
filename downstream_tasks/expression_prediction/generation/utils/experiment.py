"""High-level orchestration kept out of the command-line entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gena_expression import AnnotatedSequence, SequenceOptimizer

from .components import build_fitness, build_model, build_offspring_generator, build_scorer
from .config import RunConfig
from .inputs import (
    build_genome,
    build_plasmid_template,
    build_promoter_sampler,
    collect_promoter_population,
    load_targets,
)
from .results import write_json_file, write_results
from .resume import ResumePopulation, load_latest_population
from .snapshots import PopulationSnapshotWriter


def prepare_output_directory(
    config: dict[str, Any],
    *,
    resume_in_place: bool = False,
) -> Path:
    """Create an output directory without silently mixing independent runs."""

    output_dir = Path(config["directory"])
    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not config["allow_existing"]
        and not resume_in_place
    ):
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
    starting_population: tuple[AnnotatedSequence, ...],
    initial_generation: int,
    on_targets: dict[str, Any],
    off_targets: dict[str, Any],
    scorer: Any,
    fitness: Any,
    offspring_generator: Any,
) -> SequenceOptimizer:
    """Construct the optimizer with one starting insertion in the plasmid."""

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
        initial_population=[sample.sequence for sample in starting_population[1:]],
        initial_generation=initial_generation,
        selection_strategy=optimization["selection_strategy"],
        elite_count=optimization["elite_count"],
        tournament_size=optimization["tournament_size"],
        random_seed=optimization["random_seed"],
    )


def run_experiment(config: RunConfig) -> list[Path]:
    """Build a fresh or restored population, place it in a plasmid, and evolve."""

    output_config = config.section("output")
    resume_config = config.section("resume")
    optimization = config.section("optimization")
    sampler_config = config.section("sampler")
    resume_population: ResumePopulation | None = None
    if resume_config["from_latest_snapshot"]:
        resume_population = load_latest_population(
            resume_config["source_directory"],
            expected_sequence_length=sampler_config["seq_len"],
            expected_population_size=optimization["population_size"],
            final_generation=optimization["n_generations"],
            allow_population_shortfall=(
                resume_config["genome_sequences_to_add"] > 0
            ),
            current_config=config.as_dict(),
        )

    resume_in_place = (
        resume_population is not None
        and Path(resume_config["source_directory"]) == Path(output_config["directory"])
    )
    output_dir = prepare_output_directory(
        output_config,
        resume_in_place=resume_in_place,
    )
    config_filename = (
        "resume_config.json" if resume_population is not None else "run_config.json"
    )
    write_json_file(output_dir / config_filename, config.as_dict())

    additions: tuple[AnnotatedSequence, ...] = ()
    if resume_population is None:
        genome_sample_count = optimization["initial_population_size"]
        excluded_sequences: tuple[str, ...] = ()
    else:
        genome_sample_count = resume_config["genome_sequences_to_add"]
        excluded_sequences = resume_population.insertion_sequences

    genome_samples: tuple[AnnotatedSequence, ...] = ()
    if genome_sample_count:
        genome = build_genome(config.section("genome"))
        try:
            sampler = build_promoter_sampler(genome, sampler_config)
            genome_samples = collect_promoter_population(
                sampler,
                population_size=genome_sample_count,
                seq_len=sampler_config["seq_len"],
                unique_sequences=sampler_config["unique_sequences"],
                max_attempts=sampler_config["max_attempts"],
                excluded_sequences=excluded_sequences,
            )
        finally:
            genome.close()

    if resume_population is None:
        starting_population = genome_samples
        initial_generation = 0
    else:
        restored = tuple(
            AnnotatedSequence(
                sequence,
                name=f"restored_{index:04d}",
                metadata={
                    "source_snapshot": str(resume_population.snapshot_directory),
                    "restored_generation": resume_population.generation,
                },
            )
            for index, sequence in enumerate(
                resume_population.insertion_sequences
            )
        )
        additions = genome_samples
        starting_population = (*restored, *additions)
        initial_generation = resume_population.generation
        write_json_file(
            output_dir / "resume.json",
            resume_population.metadata(
                genome_sequences_added=len(additions),
            ),
        )

    on_targets, off_targets = load_targets(config.section("conditions"))

    template = build_plasmid_template(
        config.section("plasmid"),
        starting_population[0],
    )
    scorer = build_scorer(config.section("scorer"))
    fitness = build_fitness(config.section("fitness"))
    offspring_generator = build_offspring_generator(config.section("offspring"))
    try:
        model = build_model(config.section("model"))
        optimizer = build_optimizer(
            config,
            template=template,
            starting_population=starting_population,
            initial_generation=initial_generation,
            on_targets=on_targets,
            off_targets=off_targets,
            scorer=scorer,
            fitness=fitness,
            offspring_generator=offspring_generator,
        )
        runtime = config.section("runtime")
        snapshot_writer = PopulationSnapshotWriter.from_config(
            output_dir,
            config.section("snapshots"),
            skip_generations=(
                ()
                if resume_population is None
                else (resume_population.generation,)
            ),
        )
        optimizer.run_evolution(
            model=model,
            on_generation_complete=snapshot_writer,
            **runtime,
        )
    finally:
        close = getattr(offspring_generator, "close", None)
        if callable(close):
            close()

    return [
        *snapshot_writer.written_paths,
        *write_results(
            optimizer,
            output_dir,
            config=config,
            sampled_population=(
                starting_population
                if resume_population is None
                else additions
            ),
            resume_metadata=(
                None
                if resume_population is None
                else resume_population.metadata(
                    genome_sequences_added=len(additions),
                )
            ),
            config_filename=config_filename,
        ),
    ]
