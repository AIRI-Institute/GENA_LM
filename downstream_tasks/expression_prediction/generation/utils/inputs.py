"""Genome sampling, condition loading, and plasmid-template construction."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from gena_expression import (
    AnnotatedSequence,
    DescriptionLookup,
    Feature,
    Genome,
    PlasmidRecord,
    PromoterSampler,
)
from gena_expression.dna.plasmid import centered_plasmid_window


def load_target_group(group: Mapping[str, Any], common: Mapping[str, Any]) -> dict[str, Any]:
    """Load one on-target or off-target condition group deterministically."""

    keys = list(group["keys"])
    if not keys:
        return {}
    lookup = DescriptionLookup(
        json_dir=group["directory"],
        keys=keys,
        seed=common["seed"],
        strict=common["strict"],
        search_fields=common["search_fields"],
        filters=dict(group["filters"]),
        key_filters=dict(group["key_filters"]),
    )
    return {key: lookup.condition(key) for key in keys}


def load_targets(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load configured on-target and off-target condition mappings."""

    return (
        load_target_group(config["on_target"], config),
        load_target_group(config["off_target"], config),
    )


def build_genome(config: Mapping[str, Any]) -> Genome:
    """Construct the genome used exclusively by ``PromoterSampler``."""

    return Genome(
        fasta_path=config["fasta_path"],
        annotation=config["annotation_path"],
        build=config["build"],
        chrom_style=config["chrom_style"],
        cache=config["cache"],
    )


def build_promoter_sampler(genome: Genome, config: Mapping[str, Any]) -> PromoterSampler:
    """Construct a reproducible genomic promoter sampler."""

    return PromoterSampler(
        genome=genome,
        seq_len=config["seq_len"],
        tss_sampling_window=config["tss_sampling_window"],
        seed=config["seed"],
        transcript_feature_types=config["transcript_feature_types"],
        include_features=config["include_features"],
    )


def collect_promoter_population(
    sampler: Iterator[AnnotatedSequence],
    *,
    population_size: int,
    seq_len: int,
    unique_sequences: bool,
    max_attempts: int,
) -> tuple[AnnotatedSequence, ...]:
    """Collect the requested number of valid genomic promoter samples."""

    collected: list[AnnotatedSequence] = []
    seen: set[str] = set()
    attempts = 0
    while len(collected) < population_size and attempts < max_attempts:
        attempts += 1
        sample = next(sampler)
        if not isinstance(sample, AnnotatedSequence):
            raise TypeError("PromoterSampler must yield AnnotatedSequence objects.")
        if len(sample) != seq_len:
            raise ValueError(
                f"PromoterSampler returned length {len(sample)}; expected {seq_len}."
            )
        invalid = sorted(set(sample.sequence) - set("ACGTN"))
        if invalid:
            raise ValueError(
                "PromoterSampler returned unsupported DNA symbols: " + ", ".join(invalid)
            )
        if unique_sequences and sample.sequence in seen:
            continue
        seen.add(sample.sequence)
        collected.append(sample)

    if len(collected) != population_size:
        raise RuntimeError(
            f"Collected {len(collected)} promoter sequences after {attempts} attempts; "
            f"requested {population_size}. Increase sampler.max_attempts or disable "
            "sampler.unique_sequences."
        )
    return tuple(collected)


def _load_and_validate_plasmid(config: Mapping[str, Any]) -> PlasmidRecord:
    """Load local plasmid files and verify FASTA/GenBank sequence agreement."""

    genbank_path = Path(config["genbank_path"])
    fasta_path = config["fasta_path"]
    reporter_name = config["reporter_feature_name"]
    plasmid = PlasmidRecord.from_files(
        genbank_path=genbank_path,
        fasta_path=fasta_path,
        reporter_feature_name=reporter_name,
    )
    if fasta_path is not None:
        genbank_only = PlasmidRecord.from_files(
            genbank_path=genbank_path,
            fasta_path=None,
            reporter_feature_name=reporter_name,
        )
        if plasmid.sequence != genbank_only.sequence:
            raise ValueError(
                "The configured plasmid FASTA sequence does not match the GenBank ORIGIN sequence."
            )
    return plasmid


def _exact_feature(sequence: AnnotatedSequence, name: str) -> Feature:
    """Resolve exactly one feature by exact annotation name."""

    matches = [feature for feature in sequence.features if feature.name == name]
    if len(matches) != 1:
        available = sorted({feature.name for feature in sequence.features})
        raise ValueError(
            f"Expected exactly one plasmid feature named {name!r}, found {len(matches)}. "
            f"Available feature names: {available}"
        )
    return matches[0]


def build_plasmid_template(
    config: Mapping[str, Any],
    sampled_promoter: AnnotatedSequence,
) -> AnnotatedSequence:
    """Replace one named plasmid feature with one genome-sampled promoter.

    Genomic annotations are deliberately not copied into the plasmid. The new
    region receives one stable optimization feature, while the sample's full
    provenance remains available in template metadata and output metadata.
    """

    plasmid = _load_and_validate_plasmid(config)
    sequence = plasmid.annotated_sequence()
    replacement_feature = _exact_feature(sequence, config["replacement_feature_name"])
    reporter_tss = plasmid.reporter_tss_index()
    if replacement_feature.start <= reporter_tss < replacement_feature.end:
        raise ValueError(
            "The configured replacement feature overlaps the reporter TSS; "
            "a stable reporter-centered model context cannot be constructed."
        )

    center_name = config["tokenization_center_name"]
    center_conflicts = [
        feature
        for feature in sequence.features
        if feature.name == center_name or feature.type == center_name
    ]
    if center_conflicts:
        raise ValueError(
            f"tokenization_center_name {center_name!r} conflicts with an existing plasmid feature."
        )

    sequence = sequence.add_feature(
        name=center_name,
        start=reporter_tss,
        end=reporter_tss + 1,
        type="tokenization_center",
        source="plasmid",
    )
    region_name = config["region_feature_name"]
    replacement = AnnotatedSequence(
        sampled_promoter.sequence,
        name=region_name,
        features=(
            Feature(
                name=region_name,
                start=0,
                end=len(sampled_promoter),
                type="optimization_region",
                source="genome_sample",
            ),
        ),
    )
    edited = sequence.replace(
        start=replacement_feature.start,
        end=replacement_feature.end,
        replacement=replacement,
    ).with_metadata(
        reference_promoter_name=sampled_promoter.name,
        reference_promoter_metadata=dict(sampled_promoter.metadata),
        replaced_feature=config["replacement_feature_name"],
    )

    window_size = len(edited) if config["window_size"] == "full" else config["window_size"]
    window = centered_plasmid_window(
        sequence=edited,
        center=edited.resolve_center(center_name),
        circular=config["circular"],
        size=window_size,
    )
    _exact_feature(window, region_name)
    center_matches = [
        feature
        for feature in window.features
        if feature.name == center_name or feature.type == center_name
    ]
    if len(center_matches) != 1:
        raise ValueError(
            f"Expected exactly one tokenization center {center_name!r} after plasmid windowing."
        )
    return window
