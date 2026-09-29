"""AlphaGenome-backed sequence prediction using GENA token coordinates."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..conditions import Condition
from ..config import RetentionLike, retention_policy_for
from ..inference.batching import (
    GroupingMode,
    PairExecutionMode,
    PreprocessingBackend,
    _progress,
)
from ..inference.outputs import (
    PairPrediction,
    Prediction,
    retain_pair_prediction,
    retain_prediction,
)
from ..inference.tokenization import CenteredTokenizer, TokenizedSequence
from ..sequences import AnnotatedSequence, Feature, SequencePair

from transformers import AutoTokenizer


_ALPHAGENOME_SEQUENCE_LENGTHS = (2**14, 2**17, 2**19, 2**20)
_GROUPING_MODES = {"serial", "no_grouping", "condition", "sequence", "auto"}
_ONTOLOGY_KEYS = (
    "ontology_terms",
    "ontology_term",
    "ontology_curies",
    "ontology_curie",
    "ontology",
    "curie",
)
_CURIE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*:\S+$")


class AlphaGenomeSequenceModel:
    """Expose AlphaGenome predictions through the sequence-model API.

    The attached AlphaGenome model predicts a base-resolution track for one
    fixed-length DNA window. GENA tokenization is run independently, and every
    token receives the mean AlphaGenome value over its genomic span. The
    resulting tensor is stored in :class:`Prediction` with the same
    ``[batch, token, channel]`` layout used by the existing scorers.

    This class intentionally uses composition instead of inheriting from
    ``SequenceModel``: that class's constructor and private forward path are
    specific to the PyTorch expression checkpoint and its text tokenizer.
    """

    def __init__(
        self,
        model: Any,
        dna_tokenizer: Any,
        *,
        output_type: Any,
        dna_max_seq_len: int,
        token_len_for_fetch: int,
        num_before: int,
        organism: Any | None = None,
        alphagenome_sequence_length: int = 2**14,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach an initialized AlphaGenome model and a GENA DNA tokenizer.

        ``output_type`` may be an AlphaGenome ``OutputType`` member or its
        string name (for example, ``"DNASE"``). Conditions must identify one
        or more ontology CURIEs; see :meth:`_ontology_terms` for accepted
        fields.
        """

        if not callable(getattr(model, "predict_sequence", None)):
            raise TypeError("model must provide a callable predict_sequence method.")

        sequence_length = int(alphagenome_sequence_length)
        if sequence_length not in _ALPHAGENOME_SEQUENCE_LENGTHS:
            allowed = ", ".join(str(value) for value in _ALPHAGENOME_SEQUENCE_LENGTHS)
            raise ValueError(
                "alphagenome_sequence_length must be one of the supported "
                f"AlphaGenome lengths: {allowed}."
            )

        self.model = model
        self.dna_tokenizer = AutoTokenizer.from_pretrained(str(dna_tokenizer))
        self.dna_max_seq_len = int(dna_max_seq_len)
        self.dna_max_seq_tokens = self.dna_max_seq_len - 2
        self.token_len_for_fetch = int(token_len_for_fetch)
        self.num_before = int(num_before)
        self.output_type = self._normalize_output_type(output_type)
        self.output_name = self._output_type_name(self.output_type)
        self.organism = organism if organism is not None else self._default_organism()
        self.alphagenome_sequence_length = sequence_length
        self.provenance = {
            "backend": "alphagenome",
            "output_type": self.output_name,
            **dict(provenance or {}),
        }
        self.centered_tokenizer = CenteredTokenizer(
            dna_tokenizer=self.dna_tokenizer,
            dna_max_seq_len=self.dna_max_seq_len,
            token_len_for_fetch=self.token_len_for_fetch,
            num_before=self.num_before,
        )

    @staticmethod
    def _default_organism() -> Any:
        """Return AlphaGenome's human-organism enum without a hard dependency."""

        try:
            dna_model = importlib.import_module("alphagenome.models.dna_model")
        except ImportError as exc:
            raise ImportError(
                "AlphaGenome is required to choose the default organism. "
                "Install AlphaGenome or pass organism= explicitly."
            ) from exc
        return dna_model.Organism.HOMO_SAPIENS

    @staticmethod
    def _normalize_output_type(output_type: Any) -> Any:
        """Resolve a string output name to the installed AlphaGenome enum."""

        if not isinstance(output_type, str):
            return output_type

        enum_type = None
        for module_name in (
            "alphagenome.models.dna_output",
            "alphagenome.models.dna_client",
        ):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            enum_type = getattr(module, "OutputType", None)
            if enum_type is not None:
                break
        if enum_type is None:
            raise ImportError(
                "Could not import AlphaGenome OutputType. Install a compatible "
                "AlphaGenome package or pass an OutputType member directly."
            )

        member_name = output_type.strip().upper().replace("-", "_")
        if hasattr(enum_type, member_name):
            return getattr(enum_type, member_name)
        try:
            return enum_type(output_type.strip().lower())
        except (TypeError, ValueError) as exc:
            choices = ", ".join(member.name for member in enum_type)
            raise ValueError(
                f"Unknown AlphaGenome output_type {output_type!r}; choose {choices}."
            ) from exc

    @staticmethod
    def _output_type_name(output_type: Any) -> str:
        """Return a stable lower-case name for one output type."""

        value = getattr(output_type, "value", None)
        if isinstance(value, str):
            return value.lower()
        name = getattr(output_type, "name", None)
        if isinstance(name, str):
            return name.lower()
        return str(output_type).rsplit(".", 1)[-1].lower()

    @staticmethod
    def clear_cuda_cache() -> None:
        """Match the existing model surface; JAX device memory is model-owned."""

        return None

    def _as_condition(self, condition: Condition | str | Mapping[str, Any]) -> Condition:
        """Normalize a condition in the same way as ``SequenceModel``."""

        if isinstance(condition, Condition):
            return condition
        return Condition(name="condition", description=condition)

    @staticmethod
    def _as_sequence_list(
        sequences: AnnotatedSequence | str | Iterable[AnnotatedSequence | str],
    ) -> list[AnnotatedSequence | str]:
        """Normalize one or many sequences to a list."""

        if isinstance(sequences, (AnnotatedSequence, str)):
            return [sequences]
        return list(sequences)

    def _condition_list(
        self,
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None,
        condition: Condition | str | Mapping[str, Any] | None,
    ) -> list[Condition]:
        """Normalize exactly one of the singular or plural condition inputs."""

        if (conditions is None) == (condition is None):
            raise ValueError("Pass exactly one of condition or conditions.")
        if condition is not None:
            return [self._as_condition(condition)]
        return [self._as_condition(item) for item in conditions or []]

    def _broadcast_conditions(
        self,
        row_count: int,
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None,
        condition: Condition | str | Mapping[str, Any] | None,
    ) -> list[Condition]:
        """Broadcast one condition or validate row-wise conditions."""

        condition_list = self._condition_list(conditions, condition)
        if len(condition_list) == 1 and row_count != 1:
            return condition_list * row_count
        if len(condition_list) != row_count:
            raise ValueError("conditions must contain one item or match the number of rows.")
        return condition_list

    def _sequence_condition_rows(
        self,
        sequences: AnnotatedSequence | str | Iterable[AnnotatedSequence | str],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None,
        condition: Condition | str | Mapping[str, Any] | None,
    ) -> tuple[list[AnnotatedSequence | str], list[Condition]]:
        """Broadcast compatible sequence and condition inputs into rows."""

        sequence_list = self._as_sequence_list(sequences)
        condition_list = self._condition_list(conditions, condition)
        if not sequence_list:
            if len(condition_list) <= 1:
                return [], []
            raise ValueError("conditions must be empty when sequences is empty.")
        if len(sequence_list) == 1 and len(condition_list) > 1:
            sequence_list = sequence_list * len(condition_list)
        elif len(condition_list) == 1 and len(sequence_list) > 1:
            condition_list = condition_list * len(sequence_list)
        elif len(sequence_list) != len(condition_list):
            raise ValueError("sequences and conditions must have compatible lengths.")
        return sequence_list, condition_list

    def tokenize_sequence(
        self,
        sequence: AnnotatedSequence | str,
        *,
        center: int | str | Feature = "tss",
        strand: str = "+",
    ) -> TokenizedSequence:
        """Tokenize a sequence with the existing centered GENA tokenizer."""

        return self.centered_tokenizer.tokenize(
            sequence,
            center=center,
            strand=strand,
        )

    @staticmethod
    def _is_curie(value: Any) -> bool:
        """Return whether ``value`` is a syntactically plausible ontology CURIE."""

        return isinstance(value, str) and bool(_CURIE_PATTERN.fullmatch(value.strip()))

    @classmethod
    def _coerce_ontology_terms(cls, value: Any) -> tuple[Any, ...]:
        """Normalize one ontology value or a collection of ontology values."""

        if isinstance(value, str) or not isinstance(value, Iterable):
            terms = (value,)
        elif isinstance(value, Mapping):
            raise TypeError("An ontology term value cannot be a mapping.")
        else:
            terms = tuple(value)
        if not terms:
            raise ValueError("At least one ontology term is required.")
        for term in terms:
            if isinstance(term, str) and not cls._is_curie(term):
                raise ValueError(
                    f"Ontology term {term!r} is not a CURIE such as 'UBERON:0000178'."
                )
        return terms

    @classmethod
    def _ontology_terms(cls, condition: Condition) -> tuple[Any, ...]:
        """Extract ontology terms from one condition.

        Structured conditions may put a value under ``ontology_terms``,
        ``ontology_term``, ``ontology_curies``, ``ontology_curie``,
        ``ontology``, or ``curie`` in either ``Condition.metadata`` or a
        mapping-valued ``Condition.description``. A string description or the
        condition name is also accepted when it is itself a CURIE.
        """

        containers = [condition.metadata]
        if isinstance(condition.description, Mapping):
            containers.append(condition.description)
        for container in containers:
            if container is None:
                continue
            for key in _ONTOLOGY_KEYS:
                if key in container and container[key] is not None:
                    return cls._coerce_ontology_terms(container[key])

        if cls._is_curie(condition.description):
            return (str(condition.description).strip(),)
        if cls._is_curie(condition.name):
            return (condition.name.strip(),)
        raise ValueError(
            "AlphaGenome conditions must provide ontology CURIEs in the "
            "condition description/name or an ontology_terms-style metadata field."
        )

    def _alphagenome_window(
        self,
        tokenized: TokenizedSequence,
    ) -> tuple[str, int]:
        """Build the fixed AlphaGenome window and return its source-coordinate origin."""

        if tokenized.strand != "+":
            raise NotImplementedError(
                "AlphaGenomeSequenceModel currently supports strand='+' only; "
                "negative-strand tracks require AlphaGenome channel reindexing."
            )

        source_text = tokenized.source.sequence.upper()
        invalid = sorted(set(source_text) - set("ACGTN"))
        if invalid:
            raise ValueError(
                "AlphaGenome input contains unsupported DNA symbols: "
                + ", ".join(repr(value) for value in invalid)
            )

        length = self.alphagenome_sequence_length
        origin = int(tokenized.center) - length // 2
        source_start = min(max(origin, 0), len(source_text))
        source_end = min(max(origin + length, 0), len(source_text))
        left_pad = max(0, min(length, -origin))
        source_segment = source_text[source_start:source_end]
        right_pad = length - left_pad - len(source_segment)
        window = "N" * left_pad + source_segment + "N" * right_pad
        if len(window) != length:
            raise RuntimeError("Internal error while constructing the AlphaGenome window.")

        window_end = origin + length
        for token in tokenized.tokens:
            start = int(token["start"])
            end = int(token["end"])
            if end <= start:
                raise ValueError(f"GENA token has an empty coordinate span: {token!r}")
            if start < origin or end > window_end:
                raise ValueError(
                    "The GENA tokenized span does not fit in the AlphaGenome "
                    f"window [{origin}, {window_end}). Choose a larger "
                    "alphagenome_sequence_length."
                )
        return window, origin

    @staticmethod
    def _metadata_for_provenance(metadata: Any) -> Any:
        """Convert AlphaGenome track metadata to a useful lightweight form."""

        if metadata is None:
            return None
        if isinstance(metadata, Mapping):
            return dict(metadata)
        to_dict = getattr(metadata, "to_dict", None)
        if callable(to_dict):
            try:
                return to_dict(orient="records")
            except TypeError:
                return to_dict()
        return repr(metadata)

    def _predict_track(self, sequence: str, ontology_terms: tuple[Any, ...]) -> tuple[Any, Any]:
        """Run AlphaGenome and return one validated base-resolution track."""
        from alphagenome.models import dna_client
        output = self.model.predict_sequence(
            sequence=sequence.center(
                                    dna_client.SEQUENCE_LENGTH_16KB, 'N'
                                    ),
            organism=self.organism,
            requested_outputs=[self.output_type],
            ontology_terms=ontology_terms,
        )
        get_track = getattr(output, "get", None)
        if not callable(get_track):
            raise TypeError("AlphaGenome predict_sequence did not return an Output-like object.")
        track = get_track(self.output_type)
        if track is None:
            raise ValueError(
                f"AlphaGenome returned no {self.output_name!r} tracks for the requested ontology terms."
            )

        import numpy as np

        values = np.asarray(track.values)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(
                "AlphaGenome track values must have shape [base, channel], "
                f"got {tuple(values.shape)}."
            )
        if values.shape[0] != len(sequence):
            resolution = getattr(track, "resolution", None)
            raise ValueError(
                f"AlphaGenome output {self.output_name!r} is not base-resolution: "
                f"received {values.shape[0]} rows for {len(sequence)} bases "
                f"(reported resolution={resolution!r})."
            )
        if values.shape[1] == 0:
            raise ValueError("AlphaGenome returned a track with no channels.")
        return values, track

    @staticmethod
    def _average_over_tokens(
        values: Any,
        tokenized: TokenizedSequence,
        origin: int,
    ) -> Any:
        """Average base-level values across every GENA DNA-token span."""

        import torch

        token_count = int(tokenized.input_ids.numel())
        channel_count = int(values.shape[1])
        logits = torch.full(
            (1, token_count, channel_count),
            float("nan"),
            dtype=torch.float32,
        )
        for token in tokenized.tokens:
            input_position = int(token["input_position"])
            start = int(token["start"]) - origin
            end = int(token["end"]) - origin
            token_mean = values[start:end].mean(axis=0)
            logits[0, input_position, :] = torch.tensor(token_mean, dtype=torch.float32)
        return logits

    @staticmethod
    def _ontology_labels(terms: tuple[Any, ...]) -> list[str]:
        """Return serializable ontology labels for prediction provenance."""

        labels = []
        for term in terms:
            curie = getattr(term, "curie", None)
            labels.append(str(curie if curie is not None else term))
        return labels

    def _predict_tokenized(
        self,
        tokenized: TokenizedSequence,
        condition: Condition,
        *,
        grouping: GroupingMode,
        return_tokens: bool,
    ) -> Prediction:
        """Predict one already-tokenized row without applying retention."""

        self._validate_grouping(grouping)
        ontology_terms = self._ontology_terms(condition)
        sequence, origin = self._alphagenome_window(tokenized)
        values, track = self._predict_track(sequence, ontology_terms)
        logits = self._average_over_tokens(values, tokenized, origin)
        track_metadata = self._metadata_for_provenance(
            getattr(track, "metadata", None)
        )
        provenance = {
            **self.provenance,
            "ontology_terms": self._ontology_labels(ontology_terms),
            "alphagenome_sequence_length": self.alphagenome_sequence_length,
            "alphagenome_window_start": origin,
            "alphagenome_window_end": origin + self.alphagenome_sequence_length,
            "track_resolution": getattr(track, "resolution", 1),
            "requested_grouping": grouping,
            "resolved_grouping": "serial",
        }
        if track_metadata is not None:
            provenance["track_metadata"] = track_metadata
        return Prediction(
            sequence=tokenized.source,
            sequence_name=tokenized.source.name,
            condition=condition,
            logits=logits,
            outputs={"logits": logits},
            tokens=tokenized if return_tokens else None,
            provenance=provenance,
            description_tokens=None,
        )

    @staticmethod
    def _validate_grouping(grouping: GroupingMode) -> None:
        """Validate a grouping value while resolving execution to serial calls."""

        if grouping not in _GROUPING_MODES:
            choices = ", ".join(sorted(_GROUPING_MODES))
            raise ValueError(f"Unknown grouping mode {grouping!r}; choose {choices}.")

    @staticmethod
    def _validate_multiple_options(
        *,
        preprocessing_workers: int,
        preprocessing_backend: PreprocessingBackend,
        max_records_per_forward: int | None,
        prefetch_batches: int,
    ) -> None:
        """Validate shared batch arguments accepted for API compatibility."""

        if preprocessing_workers < 0:
            raise ValueError("preprocessing_workers must be non-negative.")
        if preprocessing_backend not in {"thread", "process"}:
            raise ValueError("preprocessing_backend must be 'thread' or 'process'.")
        if max_records_per_forward is not None and max_records_per_forward <= 0:
            raise ValueError("max_records_per_forward must be positive.")
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative.")

    def predict_sequence(
        self,
        sequence: AnnotatedSequence | str,
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        strand: str = "+",
        grouping: GroupingMode = "no_grouping",
        return_tokens: bool = True,
        retention: RetentionLike = None,
    ) -> Prediction:
        """Predict a token-averaged AlphaGenome track for one sequence."""

        condition_obj = self._as_condition(condition)
        tokenized = self.tokenize_sequence(sequence, center=center, strand=strand)
        prediction = self._predict_tokenized(
            tokenized,
            condition_obj,
            grouping=grouping,
            return_tokens=return_tokens,
        )
        return retain_prediction(
            prediction,
            retention_policy_for(retention).prediction,
        )

    def predict_multiple_sequences(
        self,
        sequences: AnnotatedSequence | str | Iterable[AnnotatedSequence | str],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        condition: Condition | str | Mapping[str, Any] | None = None,
        center: int | str | Feature = "tss",
        strand: str = "+",
        grouping: GroupingMode = "no_grouping",
        preprocessing_workers: int = 0,
        preprocessing_backend: PreprocessingBackend = "process",
        max_records_per_forward: int | None = None,
        prefetch_batches: int = 1,
        show_progress: bool = True,
        return_tokens: bool = True,
        retention: RetentionLike = None,
    ) -> list[Prediction]:
        """Predict sequence/condition rows serially with the standard call shape."""

        self._validate_grouping(grouping)
        self._validate_multiple_options(
            preprocessing_workers=preprocessing_workers,
            preprocessing_backend=preprocessing_backend,
            max_records_per_forward=max_records_per_forward,
            prefetch_batches=prefetch_batches,
        )
        sequence_list, condition_list = self._sequence_condition_rows(
            sequences,
            conditions,
            condition,
        )
        if not sequence_list:
            return []

        prediction_retention = retention_policy_for(retention).prediction
        rows = list(zip(sequence_list, condition_list))
        iterator = _progress(
            rows,
            total=len(rows),
            description="Predicting AlphaGenome sequences",
            enabled=show_progress,
            unit="item",
        )
        predictions = []
        for row_sequence, row_condition in iterator:
            tokenized = self.tokenize_sequence(
                row_sequence,
                center=center,
                strand=strand,
            )
            prediction = self._predict_tokenized(
                tokenized,
                row_condition,
                grouping=grouping,
                return_tokens=return_tokens,
            )
            predictions.append(retain_prediction(prediction, prediction_retention))
        return predictions

    def predict_pair(
        self,
        pair: SequencePair,
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        grouping: GroupingMode = "no_grouping",
        pair_execution: PairExecutionMode = "separate",
        retention: RetentionLike = None,
    ) -> PairPrediction:
        """Predict both alleles of one sequence pair."""

        return self.predict_multiple_pairs(
            [pair],
            condition=condition,
            center=center,
            grouping=grouping,
            pair_execution=pair_execution,
            prefetch_batches=0,
            show_progress=False,
            retention=retention,
        )[0]

    def predict_multiple_pairs(
        self,
        pairs: Iterable[SequencePair],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        condition: Condition | str | Mapping[str, Any] | None = None,
        center: int | str | Feature = "tss",
        grouping: GroupingMode = "no_grouping",
        pair_execution: PairExecutionMode = "separate",
        preprocessing_workers: int = 0,
        preprocessing_backend: PreprocessingBackend = "process",
        max_records_per_forward: int | None = None,
        max_pairs_per_forward: int | None = None,
        prefetch_batches: int = 1,
        show_progress: bool = True,
        retention: RetentionLike = None,
    ) -> list[PairPrediction]:
        """Predict reference/alternative pairs using separate AlphaGenome calls."""

        self._validate_grouping(grouping)
        self._validate_multiple_options(
            preprocessing_workers=preprocessing_workers,
            preprocessing_backend=preprocessing_backend,
            max_records_per_forward=max_records_per_forward,
            prefetch_batches=prefetch_batches,
        )
        if pair_execution not in {"joint", "separate"}:
            raise ValueError("pair_execution must be 'joint' or 'separate'.")
        if pair_execution == "joint":
            raise NotImplementedError(
                "AlphaGenomeSequenceModel currently supports only "
                "pair_execution='separate'."
            )
        if max_pairs_per_forward is not None:
            raise ValueError("max_pairs_per_forward requires pair_execution='joint'.")

        pair_list = list(pairs)
        condition_list = self._broadcast_conditions(
            len(pair_list),
            conditions,
            condition,
        )
        if not pair_list:
            return []

        prediction_retention = retention_policy_for(retention).prediction
        rows = list(zip(pair_list, condition_list))
        iterator = _progress(
            rows,
            total=len(rows),
            description="Predicting AlphaGenome allele pairs",
            enabled=show_progress,
            unit="pair",
        )
        predictions = []
        for pair, row_condition in iterator:
            pair_center = self._pair_center_or_variant(pair, center)
            ref_tokenized = self.tokenize_sequence(pair.ref, center=pair_center)
            alt_tokenized = self.tokenize_sequence(pair.alt, center=pair_center)
            ref_prediction = self._predict_tokenized(
                ref_tokenized,
                row_condition,
                grouping=grouping,
                return_tokens=True,
            )
            alt_prediction = self._predict_tokenized(
                alt_tokenized,
                row_condition,
                grouping=grouping,
                return_tokens=True,
            )
            full_pair = PairPrediction(
                ref=ref_prediction,
                alt=alt_prediction,
                pair=pair,
                condition=row_condition,
            )
            predictions.append(retain_pair_prediction(full_pair, prediction_retention))
        return predictions

    @staticmethod
    def _pair_center_or_variant(
        pair: SequencePair,
        center: int | str | Feature,
    ) -> int | str | Feature:
        """Match ``SequenceModel``'s TSS-to-variant fallback for pair inputs."""

        if not isinstance(center, str):
            return center
        try:
            pair.ref.resolve_center(center)
            pair.alt.resolve_center(center)
            return center
        except KeyError:
            if (
                center == "tss"
                and pair.ref.feature("variant", required=False)
                and pair.alt.feature("variant", required=False)
            ):
                return "variant"
            raise