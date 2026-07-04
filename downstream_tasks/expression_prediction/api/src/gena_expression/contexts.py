"""Context builders that turn variants into sequence pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Mapping

from .plasmids import PlasmidRecord, PrimerTails
from .sequences import AnnotatedSequence, Feature, SequencePair
from .variants import Variant


class Context(Protocol):
    """Protocol implemented by context builders."""

    def build(
        self,
        variant: Variant,
        *,
        genome: Any | None = None,
        name: str | None = None,
    ) -> SequencePair:
        """Build reference and alternative model-input sequences."""


@dataclass(frozen=True)
class GenomeContext:
    """Create a variant-centered reference/alternative pair from a genome."""

    length: int
    center: Literal["variant", "tss"] | str | int = "variant"
    strand: str = "+"
    include_features: bool = True
    pad: str = "N"

    def build(
        self,
        variant: Variant,
        *,
        genome: Any | None = None,
        name: str | None = None,
    ) -> SequencePair:
        """Build a sequence pair by fetching local genomic sequence."""

        if genome is None:
            raise ValueError("GenomeContext.build() requires genome=...")
        if variant.chrom is None or variant.pos is None:
            raise ValueError("GenomeContext requires a variant with chrom and pos.")

        if self.center == "variant":
            center = variant.pos
        elif isinstance(self.center, int):
            center = self.center
        else:
            center = int(variant.metadata.get(str(self.center), variant.pos))

        start = int(center) - self.length // 2
        ref = genome.sequence(
            variant.chrom,
            start,
            start + self.length,
            strand=self.strand,
            include_features=self.include_features,
            name=name or variant.id or "reference",
        )
        local_variant_start = variant.pos - start
        ref = ref.add_feature(
            "variant",
            local_variant_start,
            local_variant_start + max(1, len(variant.ref)),
            type="variant",
            source="variant",
            metadata=variant.to_dict(),
        )
        alt = variant.apply_to(ref, offset=local_variant_start)
        pair = SequencePair(
            ref=ref,
            alt=alt,
            variant=variant,
            metadata={"context": "GenomeContext", "length": self.length, "center": self.center},
        )
        pair.assert_compatible()
        return pair

    def sequence_at_tss(
        self,
        genome: Any,
        chrom: str,
        tss: int,
        *,
        strand: str = "+",
        name: str | None = None,
    ) -> AnnotatedSequence:
        """Fetch a sequence centered on a TSS and annotate the TSS base."""

        return genome.sequence_around(
            chrom,
            tss,
            self.length,
            strand=strand,
            include_features=self.include_features,
            center_feature_name="tss",
            name=name,
        )


@dataclass(frozen=True)
class PlasmidContext:
    """Build annotated MPRA plasmid contexts.

    Prefer passing a :class:`PlasmidRecord`, which preserves GenBank
    annotations through MCS replacement and reporter-centered windowing. A
    legacy callable shaped like ``context_fn(sequence, element) -> sequence`` is
    still accepted as a compatibility fallback, but it cannot transfer original
    plasmid annotations.
    """

    plasmid: PlasmidRecord | Any
    element: str
    primer_tails: PrimerTails | None = None
    context_size: int | None = None
    circular: bool = True
    allow_repeats: bool = False
    center_feature_name: str = "tss"
    variant_feature_name: str = "variant"
    name: str | None = None

    def build(
        self,
        variant: Variant,
        *,
        genome: Any | None = None,
        name: str | None = None,
    ) -> SequencePair:
        """Insert reference/alternative MPRA fragments into an annotated plasmid."""

        def _fragment_from_variant(
            value: Any,
            *,
            name: str,
            start_key: str,
            end_key: str,
            features_key: str,
            allele_length: int,
        ) -> AnnotatedSequence:
            """Return an annotated fragment with the variant interval marked."""

            if isinstance(value, AnnotatedSequence):
                fragment = value
            else:
                features = tuple(
                    Feature(
                        str(item["name"]),
                        int(item["start"]),
                        int(item["end"]),
                        type=str(item.get("type", "feature")),
                        strand=item.get("strand"),
                        source=item.get("source"),
                        metadata=item.get("metadata"),
                    )
                    for item in variant.metadata.get(features_key, ())
                    if isinstance(item, Mapping)
                )
                fragment = AnnotatedSequence(str(value), name=name, features=features)
            if fragment.feature(self.variant_feature_name, required=False) is not None:
                return fragment
            start = int(variant.metadata.get(start_key, variant.pos if variant.pos is not None else 0))
            end = int(variant.metadata.get(end_key, start + max(1, allele_length)))
            if len(fragment) == 0:
                start = 0
                end = 0
            else:
                start = max(0, min(start, len(fragment) - 1))
                end = min(max(start + 1, end), len(fragment))
            return fragment.add_feature(
                self.variant_feature_name,
                start,
                end,
                type=self.variant_feature_name,
                source="variant",
                metadata=variant.to_dict(),
            )

        ref_fragment = _fragment_from_variant(
            variant.metadata.get("ref_fragment", variant.metadata.get("ref_sequence", variant.ref)),
            name=f"{name or self.name or self.element}_ref_fragment",
            start_key="ref_change_start",
            end_key="ref_change_end",
            features_key="ref_features",
            allele_length=len(variant.ref),
        )
        alt_fragment = _fragment_from_variant(
            variant.metadata.get("alt_fragment", variant.metadata.get("alt_sequence", variant.alt)),
            name=f"{name or self.name or self.element}_alt_fragment",
            start_key="alt_change_start",
            end_key="alt_change_end",
            features_key="alt_features",
            allele_length=len(variant.alt),
        )

        if isinstance(self.plasmid, PlasmidRecord):
            if self.primer_tails is None:
                raise ValueError("PlasmidContext with PlasmidRecord requires primer_tails.")
            context_fn = self.plasmid.plasmid_context(
                context_size=self.context_size,
                circular=self.circular,
                allow_repeats=self.allow_repeats,
                primer_tails=self.primer_tails,
            )
            ref = context_fn(ref_fragment, self.element)
            alt = context_fn(alt_fragment, self.element)
        elif callable(self.plasmid):
            ref_raw = self.plasmid(ref_fragment, self.element)
            alt_raw = self.plasmid(alt_fragment, self.element)
            ref = ref_raw if isinstance(ref_raw, AnnotatedSequence) else AnnotatedSequence(str(ref_raw))
            alt = alt_raw if isinstance(alt_raw, AnnotatedSequence) else AnnotatedSequence(str(alt_raw))
            center = len(ref) // 2
            ref = ref.add_feature(self.center_feature_name, center, center + 1, type=self.center_feature_name)
            alt = alt.add_feature(self.center_feature_name, len(alt) // 2, len(alt) // 2 + 1, type=self.center_feature_name)
        else:
            raise TypeError("plasmid must be a PlasmidRecord or a legacy context callable.")

        ref = ref.with_metadata(context_name=name or self.name or f"{self.element}:ref")
        alt = alt.with_metadata(context_name=name or self.name or f"{self.element}:alt")
        return SequencePair(
            ref=ref,
            alt=alt,
            variant=variant,
            metadata={
                "context": "PlasmidContext",
                "element": self.element,
                "context_size": self.context_size,
                "circular": self.circular,
                "annotated_plasmid": isinstance(self.plasmid, PlasmidRecord),
            },
        )
