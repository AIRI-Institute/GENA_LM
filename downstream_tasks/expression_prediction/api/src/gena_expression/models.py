"""Model loading, prediction, and high-level variant interpretation."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from .conditions import Condition, metadata_to_description
from .predictions import ExpressionPrediction, PairPrediction, Prediction
from .sequences import AnnotatedSequence, Feature, SequencePair
from .tokenization import CenteredTokenizer, TokenizedSequence
from .variants import Variant


class SequenceModel:
    """Load tokenizers/model checkpoint and expose prediction helpers.

    The dynamic file loader, Hydra config loading, checkpoint loading,
    description formatting, description tokenization, and model call signature
    are copied from the attached benchmark/scorer code.
    """

    def __init__(
        self,
        model: Any,
        dna_tokenizer: Any,
        description_tokenizer: Any,
        *,
        dna_max_seq_len: int,
        desc_max_seq_len: int,
        token_len_for_fetch: int,
        num_before: int,
        device: str | None = None,
        output_names: Mapping[str, int | str] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        import torch

        self.logger = logging.getLogger(__name__)
        self.model = model
        self.dna_tokenizer = dna_tokenizer
        self.desc_tokenizer = description_tokenizer
        self.dna_max_seq_len = int(dna_max_seq_len)
        self.dna_max_seq_tokens = self.dna_max_seq_len - 2
        self.desc_max_seq_len = int(desc_max_seq_len)
        self.token_len_for_fetch = int(token_len_for_fetch)
        self.num_before = int(num_before)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_names = dict(output_names or {"expression": 0, "track": 0, "atac": 0})
        self.provenance = dict(provenance or {})
        self.centered_tokenizer = CenteredTokenizer(
            dna_tokenizer=self.dna_tokenizer,
            dna_max_seq_len=self.dna_max_seq_len,
            token_len_for_fetch=self.token_len_for_fetch,
            num_before=self.num_before,
            cls_id=self.dna_tokenizer.cls_token_id,
            sep_id=self.dna_tokenizer.sep_token_id,
            pad_id=self.dna_tokenizer.pad_token_id,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(
        cls,
        model_cls: str,
        checkpoint: str | Path,
        config: str | Path,
        dna_tokenizer: str | Path,
        description_tokenizer: str | Path,
        *,
        dna_max_seq_len: int,
        desc_max_seq_len: int,
        token_len_for_fetch: int,
        num_before: int,
        device: str | None = None,
        output_names: Mapping[str, int | str] | None = None,
    ) -> "SequenceModel":
        """Load model class, Hydra config, checkpoint, and tokenizers.

        ``model_cls`` accepts either ``/path/to/file.py::ClassName`` as in the
        attached code, or an import path such as ``package.module::ClassName``.
        """

        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        logger = logging.getLogger(__name__)
        model_path_or_module, class_name = model_cls.split("::")

        @contextmanager
        def _temporary_sys_path(path: Path):
            path_str = str(path)
            added = path_str not in sys.path
            if added:
                sys.path.insert(0, path_str)
            try:
                yield
            finally:
                if added:
                    sys.path.remove(path_str)

        def _load_model_class():
            maybe_path = Path(model_path_or_module)
            if maybe_path.exists() or maybe_path.suffix == ".py":
                model_path = maybe_path.resolve()
                module_name = f"_loaded_model_{model_path.stem}"
                spec = importlib.util.spec_from_file_location(module_name, model_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load module from {model_path}")

                module = importlib.util.module_from_spec(spec)
                with _temporary_sys_path(path=model_path.parent):
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                try:
                    return getattr(module, class_name), {"model_path": str(model_path), "model_class": class_name}
                except AttributeError as exc:
                    raise ImportError(f"{class_name!r} not found in {model_path}") from exc

            module = importlib.import_module(model_path_or_module)
            try:
                return getattr(module, class_name), {"model_module": model_path_or_module, "model_class": class_name}
            except AttributeError as exc:
                raise ImportError(f"{class_name!r} not found in module {model_path_or_module!r}") from exc

        logger.info("Loading model on %s", device)
        loaded_cls, model_provenance = _load_model_class()

        config_path = Path(config)
        with initialize_config_dir(str(config_path.parents[0])):
            experiment_config = compose(config_name=config_path.name)
        model_kwargs = instantiate(experiment_config["model_kwargs"])
        model = loaded_cls(**model_kwargs)

        checkpoint_path = Path(checkpoint)
        if checkpoint_path.suffix == ".tensors":
            state_dict = load_file(str(checkpoint_path), device="cpu")
        else:
            state_dict = torch.load(
                str(checkpoint_path),
                map_location=torch.device("cpu"),
                weights_only=True,
            )
        model.load_state_dict(state_dict)

        dna_tok = AutoTokenizer.from_pretrained(str(dna_tokenizer))
        desc_tok = AutoTokenizer.from_pretrained(str(description_tokenizer), padding_side="left")
        return cls(
            model=model,
            dna_tokenizer=dna_tok,
            description_tokenizer=desc_tok,
            dna_max_seq_len=dna_max_seq_len,
            desc_max_seq_len=desc_max_seq_len,
            token_len_for_fetch=token_len_for_fetch,
            num_before=num_before,
            device=device,
            output_names=output_names,
            provenance={
                **model_provenance,
                "checkpoint": str(checkpoint_path),
                "config": str(config_path),
                "dna_tokenizer": str(dna_tokenizer),
                "description_tokenizer": str(description_tokenizer),
            },
        )

    @staticmethod
    def make_description(condition: Condition | str | Mapping[str, Any]) -> str:
        """Return description text using the attached metadata sentence format."""

        if isinstance(condition, Condition):
            return condition.text()
        if isinstance(condition, str):
            return condition
        return metadata_to_description(condition)

    @staticmethod
    def make_description_from_json(meta: Mapping[str, Any]) -> str:
        """Compatibility alias for the attached benchmark helper name."""

        return metadata_to_description(meta)

    def _as_condition(self, condition: Condition | str | Mapping[str, Any]) -> Condition:
        if isinstance(condition, Condition):
            return condition
        if isinstance(condition, str):
            return Condition(name="condition", description=condition)
        return Condition(name="condition", description=condition)

    def tokenize_description(self, condition: Condition | str | Mapping[str, Any]) -> dict[str, Any]:
        """Tokenize condition text exactly like the attached code."""

        description = self.make_description(condition)
        encoding = self.desc_tokenizer(
            description,
            padding="max_length",
            padding_side="left",
            truncation=True,
            max_length=self.desc_max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        return {"desc_input_ids": input_ids, "desc_attention_mask": attention_mask}

    def tokenize_sequence(
        self,
        sequence: AnnotatedSequence | str,
        *,
        center: int | str | Feature = "tss",
        strand: str = "+",
    ) -> TokenizedSequence:
        """Tokenize a sequence around a biological center."""

        return self.centered_tokenizer.tokenize(sequence, center=center, strand=strand)

    def _run_model(
        self,
        tokenized: TokenizedSequence,
        desc_encoded: dict[str, Any],
        dataset_flag: bool | int | None = False,
    ):
        """Run the attached model call signature on one tokenized sequence."""

        import torch

        if dataset_flag is None:
            dataset_flag = False
        dataset_flag_tensor = torch.tensor(
            [[dataset_flag]],
            dtype=torch.bool,
            device=self.device,
        )
        return self.model(
            input_ids=tokenized.input_ids.unsqueeze(0).to(self.device),
            attention_mask=tokenized.attention_mask.unsqueeze(0).to(self.device),
            labels_mask=None,
            labels=None,
            return_dict=None,
            desc_input_ids=desc_encoded["desc_input_ids"].unsqueeze(0).to(self.device),
            desc_attention_mask=desc_encoded["desc_attention_mask"].unsqueeze(0).to(self.device),
            dataset_flag=dataset_flag_tensor,
        )

    def _run_tokenized_same_condition(
        self,
        tokenized_sequences: Sequence[TokenizedSequence],
        condition: Condition,
    ):
        """Run one condition against many tokenized DNA sequences.

        The expression model uses ``dataset_flag=False`` to compute the
        description embedding once and reuse it for every DNA sequence in the
        group.
        """

        import torch

        if not tokenized_sequences:
            raise ValueError("tokenized_sequences must be non-empty.")

        max_len = max(int(tokenized.input_ids.numel()) for tokenized in tokenized_sequences)
        input_rows = []
        mask_rows = []
        for tokenized in tokenized_sequences:
            input_ids = tokenized.input_ids
            attention_mask = tokenized.attention_mask
            pad_len = max_len - int(input_ids.numel())
            if pad_len > 0:
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full((pad_len,), int(self.dna_tokenizer.pad_token_id), dtype=input_ids.dtype),
                    ]
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.zeros((pad_len,), dtype=attention_mask.dtype),
                    ]
                )
            input_rows.append(input_ids)
            mask_rows.append(attention_mask)

        input_ids = torch.stack(input_rows, dim=0).to(self.device)
        attention_mask = torch.stack(mask_rows, dim=0).to(self.device)
        desc_encoded = self.tokenize_description(condition)
        n = len(tokenized_sequences)
        desc_input_ids = desc_encoded["desc_input_ids"].repeat(n, 1).unsqueeze(0).to(self.device)
        desc_attention_mask = desc_encoded["desc_attention_mask"].repeat(n, 1).unsqueeze(0).to(self.device)
        dataset_flag = torch.zeros((1, n), dtype=torch.bool, device=self.device)

        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        autocast_enabled = device_type == "cuda"
        with torch.autocast(device_type=device_type, enabled=autocast_enabled), torch.inference_mode():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_mask=None,
                labels=None,
                return_dict=None,
                desc_input_ids=desc_input_ids,
                desc_attention_mask=desc_attention_mask,
                dataset_flag=dataset_flag,
            )

    def _run_tokenized_same_sequence(
        self,
        tokenized_sequence: TokenizedSequence,
        conditions: Sequence[Condition],
    ):
        """Run one tokenized DNA sequence against many conditions.

        The expression model uses ``dataset_flag=True`` to compute the DNA
        representation once and reuse it for every description in the group.
        This mirrors :meth:`_run_tokenized_same_condition`, but swaps the
        repeated side of the model input.
        """

        import torch

        if not conditions:
            raise ValueError("conditions must be non-empty.")

        desc_encoded = [self.tokenize_description(condition) for condition in conditions]
        desc_input_ids = torch.stack(
            [encoded["desc_input_ids"] for encoded in desc_encoded],
            dim=0,
        ).unsqueeze(0).to(self.device)
        desc_attention_mask = torch.stack(
            [encoded["desc_attention_mask"] for encoded in desc_encoded],
            dim=0,
        ).unsqueeze(0).to(self.device)

        n = len(conditions)
        # The model still expects B*N physical rows; dataset_flag=True only
        # tells it that these DNA rows are duplicates and can be encoded once.
        input_ids = tokenized_sequence.input_ids.unsqueeze(0).repeat(n, 1).to(self.device)
        attention_mask = tokenized_sequence.attention_mask.unsqueeze(0).repeat(n, 1).to(self.device)
        dataset_flag = torch.ones((1, n), dtype=torch.bool, device=self.device)
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        autocast_enabled = device_type == "cuda"
        with torch.autocast(device_type=device_type, enabled=autocast_enabled), torch.inference_mode():
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels_mask=None,
                labels=None,
                return_dict=None,
                desc_input_ids=desc_input_ids,
                desc_attention_mask=desc_attention_mask,
                dataset_flag=dataset_flag,
            )

    @staticmethod
    def _expression_from_logits(logits: Any) -> Any:
        """Extract expression output using the attached first-token convention."""

        return logits[:, 0:1, :].squeeze(-1)

    def predict(
        self,
        sequence: AnnotatedSequence | str,
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        strand: str = "+",
        dataset_flag: bool | int | None = None,
        return_tokens: bool = True,
    ) -> Prediction:
        """Run description-conditioned prediction for one sequence."""

        import torch

        condition_obj = self._as_condition(condition)
        desc_encoded = self.tokenize_description(condition_obj)
        tokenized = self.tokenize_sequence(sequence, center=center, strand=strand)
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        autocast_enabled = device_type == "cuda"
        with torch.autocast(device_type=device_type, enabled=autocast_enabled), torch.inference_mode():
            output = self._run_model(tokenized, desc_encoded, dataset_flag=dataset_flag)
        logits = output.logits.detach().cpu()
        expression = self._expression_from_logits(logits)
        return Prediction(
            sequence=tokenized.source,
            condition=condition_obj,
            logits=logits,
            outputs={"expression": expression, "logits": logits},
            tokens=tokenized if return_tokens else None,
            provenance=self.provenance,
        )

    def predict_expression(
        self,
        sequence: AnnotatedSequence | str,
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        strand: str = "+",
        dataset_flag: bool | int | None = None,
    ) -> ExpressionPrediction:
        """Predict expression from a sequence using first-token logits."""

        prediction = self.predict(
            sequence,
            condition=condition,
            center=center,
            strand=strand,
            dataset_flag=dataset_flag,
            return_tokens=True,
        )
        expression = prediction.scalar("expression")
        return ExpressionPrediction(
            sequence=prediction.sequence,
            condition=prediction.condition,
            logits=prediction.logits,
            outputs=prediction.outputs,
            tokens=prediction.tokens,
            provenance=prediction.provenance,
            expression=expression,
        )

    def predict_many_sequences_one_condition(
        self,
        sequences: Iterable[AnnotatedSequence | str],
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        strand: str = "+",
        batch_size: int | None = None,
        return_tokens: bool = True,
    ) -> list[Prediction]:
        """Predict many sequences under one condition using batched inference.

        This is the efficient path for repeated descriptions: the model receives
        a ``dataset_flag`` group of ``False`` values, so it computes the
        description embedding once and reuses it for every DNA sequence.
        """

        condition_obj = self._as_condition(condition)
        sequence_list = list(sequences)
        if not sequence_list:
            return []
        if batch_size is None:
            batch_size = len(sequence_list)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        predictions: list[Prediction] = []
        for start in range(0, len(sequence_list), batch_size):
            chunk = sequence_list[start : start + batch_size]
            tokenized = [self.tokenize_sequence(sequence, center=center, strand=strand) for sequence in chunk]
            output = self._run_tokenized_same_condition(tokenized, condition_obj)
            logits = output.logits.detach().cpu()
            for idx, tokenized_sequence in enumerate(tokenized):
                row_logits = logits[idx : idx + 1]
                expression = self._expression_from_logits(row_logits)
                predictions.append(
                    Prediction(
                        sequence=tokenized_sequence.source,
                        condition=condition_obj,
                        logits=row_logits,
                        outputs={"expression": expression, "logits": row_logits},
                        tokens=tokenized_sequence if return_tokens else None,
                        provenance={**self.provenance, "batch_mode": "many_sequences_one_condition"},
                    )
                )
        return predictions

    def predict_one_sequence_many_conditions(
        self,
        sequence: AnnotatedSequence | str,
        conditions: Iterable[Condition | str | Mapping[str, Any]],
        *,
        center: int | str | Feature = "tss",
        strand: str = "+",
        batch_size: int | None = None,
        return_tokens: bool = True,
    ) -> list[Prediction]:
        """Predict one sequence under many conditions using batched inference.

        This is the efficient path for repeated DNA: the model receives a
        ``dataset_flag`` group of ``True`` values, so it computes the DNA
        representation once and reuses it for every description in the group.

        The optimization relies on the attached expression model's
        ``dataset_flag=True`` convention. Results are returned in the same order
        as ``conditions``.
        """

        condition_list = [self._as_condition(condition) for condition in conditions]
        if not condition_list:
            return []
        if batch_size is None:
            batch_size = len(condition_list)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        tokenized = self.tokenize_sequence(sequence, center=center, strand=strand)
        predictions: list[Prediction] = []
        for start in range(0, len(condition_list), batch_size):
            chunk = condition_list[start : start + batch_size]
            output = self._run_tokenized_same_sequence(tokenized, chunk)
            logits = output.logits.detach().cpu()
            if int(logits.shape[0]) != len(chunk):
                raise RuntimeError(
                    "Unexpected logits shape for one-sequence/many-conditions batch: "
                    f"expected first dimension {len(chunk)}, got {tuple(logits.shape)}."
                )
            for idx, condition in enumerate(chunk):
                row_logits = logits[idx : idx + 1]
                expression = self._expression_from_logits(row_logits)
                predictions.append(
                    Prediction(
                        sequence=tokenized.source,
                        condition=condition,
                        logits=row_logits,
                        outputs={"expression": expression, "logits": row_logits},
                        tokens=tokenized if return_tokens else None,
                        provenance={**self.provenance, "batch_mode": "one_sequence_many_conditions"},
                    )
                )
        return predictions

    def predict_pair(
        self,
        pair: SequencePair,
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        dataset_flag: bool | int | None = None,
    ) -> PairPrediction:
        """Predict reference and alternative sequences for a sequence pair."""

        condition_obj = self._as_condition(condition)
        center = self._pair_center_or_variant(pair, center)
        ref_prediction = self.predict(
            pair.ref,
            condition=condition_obj,
            center=center,
            strand="+",
            dataset_flag=dataset_flag,
            return_tokens=True,
        )
        alt_prediction = self.predict(
            pair.alt,
            condition=condition_obj,
            center=center,
            strand="+",
            dataset_flag=dataset_flag,
            return_tokens=True,
        )
        return PairPrediction(ref=ref_prediction, alt=alt_prediction, pair=pair, condition=condition_obj)

    def predict_many_pairs_one_condition(
        self,
        pairs: Iterable[SequencePair],
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        batch_size: int | None = None,
    ) -> list[PairPrediction]:
        """Predict many reference/alternative pairs under one condition."""

        condition_obj = self._as_condition(condition)
        pair_list = list(pairs)
        if not pair_list:
            return []

        centers = [self._pair_center_or_variant(pair, center) for pair in pair_list]
        ref_tokenized = [
            self.tokenize_sequence(pair.ref, center=pair_center, strand="+")
            for pair, pair_center in zip(pair_list, centers)
        ]
        alt_tokenized = [
            self.tokenize_sequence(pair.alt, center=pair_center, strand="+")
            for pair, pair_center in zip(pair_list, centers)
        ]

        ref_predictions = self._predict_tokenized_many_same_condition(
            ref_tokenized,
            condition_obj,
            batch_size=batch_size,
            batch_mode="many_pair_refs_one_condition",
        )
        alt_predictions = self._predict_tokenized_many_same_condition(
            alt_tokenized,
            condition_obj,
            batch_size=batch_size,
            batch_mode="many_pair_alts_one_condition",
        )
        return [
            PairPrediction(ref=ref_pred, alt=alt_pred, pair=pair, condition=condition_obj)
            for pair, ref_pred, alt_pred in zip(pair_list, ref_predictions, alt_predictions)
        ]

    def _predict_tokenized_many_same_condition(
        self,
        tokenized_sequences: Sequence[TokenizedSequence],
        condition: Condition,
        *,
        batch_size: int | None = None,
        batch_mode: str,
    ) -> list[Prediction]:
        """Convert tokenized batch outputs into :class:`Prediction` objects."""

        tokenized_list = list(tokenized_sequences)
        if not tokenized_list:
            return []
        if batch_size is None:
            batch_size = len(tokenized_list)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        predictions: list[Prediction] = []
        for start in range(0, len(tokenized_list), batch_size):
            chunk = tokenized_list[start : start + batch_size]
            output = self._run_tokenized_same_condition(chunk, condition)
            logits = output.logits.detach().cpu()
            for idx, tokenized_sequence in enumerate(chunk):
                row_logits = logits[idx : idx + 1]
                expression = self._expression_from_logits(row_logits)
                predictions.append(
                    Prediction(
                        sequence=tokenized_sequence.source,
                        condition=condition,
                        logits=row_logits,
                        outputs={"expression": expression, "logits": row_logits},
                        tokens=tokenized_sequence,
                        provenance={**self.provenance, "batch_mode": batch_mode},
                    )
                )
        return predictions

    @staticmethod
    def _pair_center_or_variant(
        pair: SequencePair,
        center: int | str | Feature,
    ) -> int | str | Feature:
        """Use the requested center, with a small variant-only fallback.

        The API doc defaults pair prediction to ``center="tss"``, but a common
        variant-only context has no TSS feature. Falling back to ``variant`` in
        that exact case makes the documented GenomeContext example usable while
        still raising for other missing centers.
        """

        if not isinstance(center, str):
            return center
        try:
            pair.ref.resolve_center(center)
            pair.alt.resolve_center(center)
            return center
        except KeyError:
            if center == "tss" and pair.ref.feature("variant", required=False) and pair.alt.feature("variant", required=False):
                return "variant"
            raise

    def predict_expression_at_tss(
        self,
        genome: Any,
        chrom: str,
        tss: int,
        *,
        condition: Condition | str | Mapping[str, Any],
        strand: str = "+",
        context_bp: int | None = None,
        coordinate_system: Literal["0-based", "1-based"] = "0-based",
        include_features: bool = True,
        center_feature_name: str = "tss",
        dataset_flag: bool | int | None = None,
        return_sequence: bool = True,
    ) -> ExpressionPrediction:
        """Predict expression directly from a genome and TSS coordinate."""

        tss0 = int(tss) - 1 if coordinate_system == "1-based" else int(tss)
        if context_bp is None:
            context_bp = 2 * self.token_len_for_fetch * max(
                self.num_before,
                self.dna_max_seq_tokens - self.num_before,
            )
        sequence = genome.sequence_around(
            chrom,
            tss0,
            int(context_bp),
            strand=strand,
            include_features=include_features,
            center_feature_name=center_feature_name,
            name=f"{chrom}:{tss0}:{strand}",
        )
        prediction = self.predict_expression(
            sequence,
            condition=condition,
            center=center_feature_name,
            strand="+",
            dataset_flag=dataset_flag,
        )
        if not return_sequence:
            prediction.sequence = None
        return prediction

    @staticmethod
    def _record_field(record: Any, field: str, default: Any = None) -> Any:
        """Return ``field`` from a mapping or row-like object."""

        if isinstance(record, Mapping):
            return record.get(field, default)
        return getattr(record, field, default)

    def predict_expression_at_tss_many(
        self,
        records: Iterable[Any],
        conditions: Iterable[Condition | str | Mapping[str, Any]],
        *,
        genome: Any,
        context_bp: int | None = None,
        coordinate_system: Literal["0-based", "1-based"] = "0-based",
        include_features: bool = True,
        center_feature_name: str = "tss",
        batch_size: int | None = None,
        return_sequence: bool = True,
    ) -> list[ExpressionPrediction]:
        """Predict expression for paired TSS records and conditions.

        ``records`` and ``conditions`` are paired row-wise and must have the
        same length. Repeated conditions are grouped internally so the model can
        reuse description embeddings efficiently while results are returned in
        the original input order.
        """

        record_list = list(records)
        condition_list = [self._as_condition(condition) for condition in conditions]
        if len(record_list) != len(condition_list):
            raise ValueError("records and conditions must have the same length.")
        if not record_list:
            return []

        sequences: list[AnnotatedSequence] = []
        for idx, record in enumerate(record_list):
            chrom = self._record_field(record, "chrom")
            tss = self._record_field(record, "tss")
            strand = self._record_field(record, "strand", "+")
            name = self._record_field(record, "name", None)
            if chrom is None or tss is None:
                raise ValueError(f"TSS record {idx} must contain chrom and tss fields.")
            tss0 = int(tss) - 1 if coordinate_system == "1-based" else int(tss)
            if context_bp is None:
                size = 2 * self.token_len_for_fetch * max(
                    self.num_before,
                    self.dna_max_seq_tokens - self.num_before,
                )
            else:
                size = int(context_bp)
            sequences.append(
                genome.sequence_around(
                    str(chrom),
                    tss0,
                    size,
                    strand=strand,
                    include_features=include_features,
                    center_feature_name=center_feature_name,
                    name=name or f"{chrom}:{tss0}:{strand}",
                )
            )

        grouped: dict[str, list[int]] = {}
        for idx, condition in enumerate(condition_list):
            grouped.setdefault(condition.text(), []).append(idx)

        output: list[ExpressionPrediction | None] = [None] * len(record_list)
        for indices in grouped.values():
            condition = condition_list[indices[0]]
            group_sequences = [sequences[idx] for idx in indices]
            group_predictions = self.predict_many_sequences_one_condition(
                group_sequences,
                condition=condition,
                center=center_feature_name,
                strand="+",
                batch_size=batch_size,
                return_tokens=True,
            )
            for idx, prediction in zip(indices, group_predictions):
                expression_prediction = ExpressionPrediction(
                    sequence=prediction.sequence,
                    condition=prediction.condition,
                    logits=prediction.logits,
                    outputs=prediction.outputs,
                    tokens=prediction.tokens,
                    provenance={
                        **dict(prediction.provenance or {}),
                        "batch_method": "predict_expression_at_tss_many",
                        "record_index": idx,
                    },
                    expression=prediction.scalar("expression"),
                )
                if not return_sequence:
                    expression_prediction.sequence = None
                output[idx] = expression_prediction

        return [prediction for prediction in output if prediction is not None]


class VariantInterpreter:
    """High-level object for building contexts, predicting variants, and scoring effects."""

    def __init__(self, model: SequenceModel) -> None:
        self.model = model

    def predict_variant(
        self,
        variant: Variant,
        *,
        context: Any,
        condition: Condition | str | Mapping[str, Any],
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        dataset_flag: bool | int | None = None,
    ) -> PairPrediction:
        """Build a sequence pair for ``variant`` and run the model."""

        pair = context.build(variant, genome=genome)
        return self.model.predict_pair(pair, condition=condition, center=center, dataset_flag=dataset_flag)

    def score_variant(
        self,
        variant: Variant,
        *,
        context: Any,
        condition: Condition | str | Mapping[str, Any],
        scorer: Any,
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        dataset_flag: bool | int | None = None,
    ):
        """Predict and score one variant."""

        prediction = self.predict_variant(
            variant,
            context=context,
            condition=condition,
            genome=genome,
            center=center,
            dataset_flag=dataset_flag,
        )
        return scorer.score(prediction)

    def score_sequence_pair(
        self,
        pair: SequencePair,
        *,
        condition: Condition | str | Mapping[str, Any],
        scorer: Any,
        center: int | str | Feature = "tss",
        dataset_flag: bool | int | None = None,
    ):
        """Predict and score an already materialized sequence pair."""

        prediction = self.model.predict_pair(
            pair,
            condition=condition,
            center=center,
            dataset_flag=dataset_flag,
        )
        return scorer.score(prediction)

    def predict_sequence_pairs_many(
        self,
        pairs: Iterable[SequencePair],
        conditions: Iterable[Condition | str | Mapping[str, Any]],
        *,
        center: int | str | Feature = "tss",
        batch_size: int | None = None,
    ) -> list[PairPrediction]:
        """Predict paired sequence pairs and conditions with internal grouping.

        ``pairs`` and ``conditions`` are paired row-wise and must have equal
        length. Repeated conditions are grouped so the model can compute one
        description embedding per group and reuse it for many DNA inputs.
        """

        pair_list = list(pairs)
        condition_list = [self.model._as_condition(condition) for condition in conditions]
        if len(pair_list) != len(condition_list):
            raise ValueError("pairs and conditions must have the same length.")
        if not pair_list:
            return []

        grouped: dict[str, list[int]] = {}
        for idx, condition in enumerate(condition_list):
            grouped.setdefault(condition.text(), []).append(idx)

        output: list[PairPrediction | None] = [None] * len(pair_list)
        for indices in grouped.values():
            condition = condition_list[indices[0]]
            group_pairs = [pair_list[idx] for idx in indices]
            group_predictions = self.model.predict_many_pairs_one_condition(
                group_pairs,
                condition=condition,
                center=center,
                batch_size=batch_size,
            )
            for idx, prediction in zip(indices, group_predictions):
                output[idx] = prediction

        return [prediction for prediction in output if prediction is not None]

    def score_sequence_pairs_many(
        self,
        pairs: Iterable[SequencePair],
        conditions: Iterable[Condition | str | Mapping[str, Any]],
        *,
        scorer: Any,
        center: int | str | Feature = "tss",
        batch_size: int | None = None,
    ) -> list[Any]:
        """Predict and score paired sequence pairs and conditions."""

        predictions = self.predict_sequence_pairs_many(
            pairs,
            conditions,
            center=center,
            batch_size=batch_size,
        )
        return [scorer.score(prediction) for prediction in predictions]

    @staticmethod
    def _variant_record_name(record: Any, fallback: str) -> str:
        """Return a display name for a variant record."""

        if isinstance(record, Mapping):
            return str(record.get("name") or record.get("id") or record.get("variant") or fallback)
        return str(getattr(record, "name", getattr(record, "id", fallback)) or fallback)

    @staticmethod
    def _variant_from_record(
        record: Any,
        *,
        genome: Any | None = None,
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
    ) -> Variant:
        """Normalize a variant object/string/table row to :class:`Variant`."""

        if isinstance(record, Variant):
            return record
        if isinstance(record, str):
            return Variant.from_str(record, genome=genome, coordinate_system=coordinate_system)
        if isinstance(record, Mapping):
            if "variant" in record:
                value = record["variant"]
                if isinstance(value, Variant):
                    return value
                return Variant.from_str(str(value), genome=genome, coordinate_system=coordinate_system)
            if {"chrom", "pos", "ref", "alt"}.issubset(record):
                variant = Variant(
                    chrom=str(record["chrom"]),
                    pos=int(record["pos"]) - 1 if coordinate_system == "1-based" else int(record["pos"]),
                    ref=str(record["ref"]),
                    alt=str(record["alt"]),
                    id=str(record.get("name") or record.get("id") or ""),
                )
                if genome is not None:
                    variant.validate(genome)
                return variant

        value = getattr(record, "variant", None)
        if value is not None:
            if isinstance(value, Variant):
                return value
            return Variant.from_str(str(value), genome=genome, coordinate_system=coordinate_system)
        chrom = getattr(record, "chrom", None)
        pos = getattr(record, "pos", None)
        ref = getattr(record, "ref", None)
        alt = getattr(record, "alt", None)
        if chrom is not None and pos is not None and ref is not None and alt is not None:
            variant = Variant(
                chrom=str(chrom),
                pos=int(pos) - 1 if coordinate_system == "1-based" else int(pos),
                ref=str(ref),
                alt=str(alt),
                id=str(getattr(record, "name", getattr(record, "id", "")) or ""),
            )
            if genome is not None:
                variant.validate(genome)
            return variant
        raise TypeError("Variant records must be Variant objects, strings, or rows with variant/chrom/pos/ref/alt fields.")

    def predict_variants_many(
        self,
        records: Iterable[Any],
        conditions: Iterable[Condition | str | Mapping[str, Any]],
        *,
        context: Any | None = None,
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        batch_size: int | None = None,
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
        **sequence_pair_kwargs: Any,
    ) -> list[PairPrediction]:
        """Predict paired variant records and conditions with internal grouping.

        ``records`` and ``conditions`` are paired row-wise and must have equal
        length. Repeated conditions are batched together using the model's
        repeated-description mode.
        """

        record_list = list(records)
        condition_list = [self.model._as_condition(condition) for condition in conditions]
        if len(record_list) != len(condition_list):
            raise ValueError("records and conditions must have the same length.")
        if not record_list:
            return []

        pairs: list[SequencePair] = []
        for idx, record in enumerate(record_list):
            variant = self._variant_from_record(record, genome=genome, coordinate_system=coordinate_system)
            name = self._variant_record_name(record, fallback=variant.id or f"variant_{idx}")
            pairs.append(
                variant.to_sequence_pair(
                    context=context,
                    genome=genome,
                    name=name,
                    **sequence_pair_kwargs,
                )
            )

        return self.predict_sequence_pairs_many(
            pairs,
            condition_list,
            center=center,
            batch_size=batch_size,
        )

    def score_variants_many(
        self,
        records: Iterable[Any],
        conditions: Iterable[Condition | str | Mapping[str, Any]],
        *,
        context: Any | None = None,
        scorer: Any,
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        batch_size: int | None = None,
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
        **sequence_pair_kwargs: Any,
    ) -> list[Any]:
        """Predict and score paired variant records and conditions."""

        predictions = self.predict_variants_many(
            records,
            conditions,
            context=context,
            genome=genome,
            center=center,
            batch_size=batch_size,
            coordinate_system=coordinate_system,
            **sequence_pair_kwargs,
        )
        return [scorer.score(prediction) for prediction in predictions]

    def score_variants(
        self,
        variants: Any,
        *,
        context: Any,
        condition: Condition | str | Mapping[str, Any],
        scorer: Any,
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        dataset_flag: bool | int | None = None,
        on_error: Literal["raise", "warn", "skip"] = "raise",
    ) -> list[Any]:
        """Score many variants, with optional warning/skip behavior on failures."""

        import warnings

        results = []
        for variant in variants:
            try:
                results.append(
                    self.score_variant(
                        variant,
                        context=context,
                        condition=condition,
                        scorer=scorer,
                        genome=genome,
                        center=center,
                        dataset_flag=dataset_flag,
                    )
                )
            except Exception as exc:
                if on_error == "raise":
                    raise
                if on_error == "warn":
                    warnings.warn(f"Skipping variant {variant}: {exc}", RuntimeWarning, stacklevel=2)
        return results
