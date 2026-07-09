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
from .results import ScoringResult, VariantReport
from .sequences import AnnotatedSequence, Feature, SequencePair
from .tokenization import CenteredTokenizer, TokenizedSequence
from .variants import Variant


_TSS_WORKER_GENOME = None
_TSS_WORKER_TOKENIZER = None


def _tss_worker_init(genome_config: Mapping[str, Any], tokenizer_config: Mapping[str, Any]) -> None:
    """Initialize per-process genome/tokenizer state for TSS preprocessing."""

    from transformers import AutoTokenizer

    from .genome import Genome

    global _TSS_WORKER_GENOME, _TSS_WORKER_TOKENIZER
    _TSS_WORKER_GENOME = Genome(**dict(genome_config))
    dna_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_config["name_or_path"]), trust_remote_code=True)
    _TSS_WORKER_TOKENIZER = CenteredTokenizer(
        dna_tokenizer=dna_tokenizer,
        dna_max_seq_len=int(tokenizer_config["dna_max_seq_len"]),
        token_len_for_fetch=int(tokenizer_config["token_len_for_fetch"]),
        num_before=int(tokenizer_config["num_before"]),
        cls_id=tokenizer_config.get("cls_id"),
        sep_id=tokenizer_config.get("sep_id"),
        pad_id=tokenizer_config.get("pad_id"),
    )


def _tss_worker_tokenize(task: Mapping[str, Any]) -> tuple[int, TokenizedSequence]:
    """Build and tokenize one TSS-centered sequence inside a worker process."""

    if _TSS_WORKER_GENOME is None or _TSS_WORKER_TOKENIZER is None:
        raise RuntimeError("TSS preprocessing worker was not initialized.")
    sequence = _TSS_WORKER_GENOME.sequence_around(
        str(task["chrom"]),
        int(task["tss0"]),
        int(task["size"]),
        strand=str(task["strand"]),
        include_features=bool(task["include_features"]),
        center_feature_name=task["center_feature_name"],
        name=task["name"],
    )
    tokenized = _TSS_WORKER_TOKENIZER.tokenize(sequence, center=task["center_feature_name"], strand="+")
    return int(task["index"]), tokenized


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
            padding=False,
            truncation=True,
            max_length=self.desc_max_seq_len,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        return {
            "description": description,
            "desc_input_ids": input_ids,
            "desc_attention_mask": attention_mask,
        }

    def tokenize_sequence(
        self,
        sequence: AnnotatedSequence | str,
        *,
        center: int | str | Feature = "tss",
        strand: str = "+",
    ) -> TokenizedSequence:
        """Tokenize a sequence around a biological center."""

        return self.centered_tokenizer.tokenize(sequence, center=center, strand=strand)

    def _stack_tokenized_dna(self, tokenized_sequences: Sequence[TokenizedSequence]):
        """Pad tokenized DNA rows to one length and move them to the model device."""

        import torch

        max_len = max(int(tokenized.input_ids.numel()) for tokenized in tokenized_sequences)
        dna_pad_id = self.dna_tokenizer.pad_token_id
        if dna_pad_id is None:
            dna_pad_id = 0
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
                        torch.full((pad_len,), int(dna_pad_id), dtype=input_ids.dtype),
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
        return torch.stack(input_rows, dim=0).to(self.device), torch.stack(mask_rows, dim=0).to(self.device)

    def _stack_description_encodings(self, encoded_descriptions: Sequence[dict[str, Any]]):
        """Pad description rows within a forward call and move them to the model device."""

        import torch

        max_desc_len = max(int(encoded["desc_input_ids"].numel()) for encoded in encoded_descriptions)
        pad_id = self.desc_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        left_pad = getattr(self.desc_tokenizer, "padding_side", "right") == "left"
        input_rows = []
        mask_rows = []
        for encoded in encoded_descriptions:
            input_row = encoded["desc_input_ids"]
            mask_row = encoded["desc_attention_mask"]
            pad_len = max_desc_len - int(input_row.numel())
            if pad_len > 0:
                input_pad = torch.full((pad_len,), int(pad_id), dtype=input_row.dtype)
                mask_pad = torch.zeros((pad_len,), dtype=mask_row.dtype)
                if left_pad:
                    input_row = torch.cat([input_pad, input_row])
                    mask_row = torch.cat([mask_pad, mask_row])
                else:
                    input_row = torch.cat([input_row, input_pad])
                    mask_row = torch.cat([mask_row, mask_pad])
            input_rows.append(input_row)
            mask_rows.append(mask_row)
        return torch.stack(input_rows, dim=0).to(self.device), torch.stack(mask_rows, dim=0).to(self.device)

    def _model_autocast_device(self) -> tuple[str, bool]:
        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return device_type, device_type == "cuda"

    @staticmethod
    def _map_preprocessing(func: Any, items: Sequence[Any], preprocessing_workers: int) -> list[Any]:
        if preprocessing_workers < 0:
            raise ValueError("preprocessing_workers must be non-negative.")
        if preprocessing_workers <= 1 or len(items) < 2:
            return [func(item) for item in items]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(preprocessing_workers)) as executor:
            return list(executor.map(func, items))

    @staticmethod
    def _as_sequence_list(sequences: AnnotatedSequence | str | Iterable[AnnotatedSequence | str]) -> list[AnnotatedSequence | str]:
        if isinstance(sequences, (AnnotatedSequence, str)):
            return [sequences]
        return list(sequences)

    def _condition_list(
        self,
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None,
        condition: Condition | str | Mapping[str, Any] | None,
    ) -> list[Condition]:
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

    @staticmethod
    def _sequence_task_key(sequence: AnnotatedSequence | str, center: int | str | Feature, strand: str) -> tuple[Any, ...]:
        sequence_key = ("str", sequence) if isinstance(sequence, str) else ("object", id(sequence))
        center_key = ("value", center) if isinstance(center, (int, str)) else ("object", id(center))
        return sequence_key, center_key, strand

    def _tokenize_sequence_tasks(
        self,
        tasks: Sequence[tuple[AnnotatedSequence | str, int | str | Feature, str]],
        *,
        preprocessing_workers: int = 0,
    ) -> list[TokenizedSequence]:
        task_list = list(tasks)
        unique_tasks: list[tuple[AnnotatedSequence | str, int | str | Feature, str]] = []
        unique_keys: list[tuple[Any, ...]] = []
        seen: set[tuple[Any, ...]] = set()
        row_keys = []
        for sequence, center, strand in task_list:
            key = self._sequence_task_key(sequence, center, strand)
            row_keys.append(key)
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
                unique_tasks.append((sequence, center, strand))

        def tokenize(task: tuple[AnnotatedSequence | str, int | str | Feature, str]) -> TokenizedSequence:
            sequence, center, strand = task
            return self.tokenize_sequence(sequence, center=center, strand=strand)

        tokenized_unique = self._map_preprocessing(tokenize, unique_tasks, preprocessing_workers)
        tokenized_by_key = dict(zip(unique_keys, tokenized_unique))
        return [tokenized_by_key[key] for key in row_keys]

    def _tokenize_descriptions(
        self,
        conditions: Sequence[Condition],
        *,
        preprocessing_workers: int = 0,
    ) -> list[dict[str, Any]]:
        unique_conditions: list[Condition] = []
        unique_keys: list[str] = []
        seen: set[str] = set()
        row_keys = []
        for condition in conditions:
            key = self._grouping_key_for_condition(condition)
            row_keys.append(key)
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
                unique_conditions.append(condition)
        encoded_unique = self._map_preprocessing(self.tokenize_description, unique_conditions, preprocessing_workers)
        encoded_by_key = dict(zip(unique_keys, encoded_unique))
        return [encoded_by_key[key] for key in row_keys]

    @staticmethod
    def _description_token_payload(desc_encoded: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "description": desc_encoded.get("description"),
            "input_ids": desc_encoded["desc_input_ids"],
            "attention_mask": desc_encoded["desc_attention_mask"],
            "desc_input_ids": desc_encoded["desc_input_ids"],
            "desc_attention_mask": desc_encoded["desc_attention_mask"],
        }

    def _attach_description_tokens(self, prediction: Prediction, desc_encoded: Mapping[str, Any]) -> Prediction:
        payload = self._description_token_payload(desc_encoded)
        setattr(prediction, "description_tokens", payload)
        return prediction

    @staticmethod
    def _copy_description_tokens(source: Prediction, target: Prediction) -> Prediction:
        payload = getattr(source, "description_tokens", None)
        if payload is not None:
            setattr(target, "description_tokens", payload)
        return target

    def _to_expression_prediction(self, prediction: Prediction, *, return_sequence: bool = True) -> ExpressionPrediction:
        expression_prediction = ExpressionPrediction(
            sequence=prediction.sequence if return_sequence else None,
            condition=prediction.condition,
            logits=prediction.logits,
            outputs=prediction.outputs,
            tokens=prediction.tokens,
            provenance=prediction.provenance,
            expression=prediction.scalar("expression"),
        )
        return self._copy_description_tokens(prediction, expression_prediction)

    def _run_tokenized_no_grouping(
        self,
        tokenized_sequences: Sequence[TokenizedSequence],
        desc_encodings: Sequence[dict[str, Any]],
    ):
        """Run row-wise pairs with ``B=len(rows), N=1`` and no duplicate shortcut."""

        import torch

        if not tokenized_sequences:
            raise ValueError("tokenized_sequences must be non-empty.")
        if len(tokenized_sequences) != len(desc_encodings):
            raise ValueError("tokenized_sequences and desc_encodings must have the same length.")

        input_ids, attention_mask = self._stack_tokenized_dna(tokenized_sequences)
        desc_input_ids, desc_attention_mask = self._stack_description_encodings(desc_encodings)
        desc_input_ids = desc_input_ids.unsqueeze(1)
        desc_attention_mask = desc_attention_mask.unsqueeze(1)
        dataset_flag = torch.zeros((len(tokenized_sequences), 1), dtype=torch.bool, device=self.device)

        device_type, autocast_enabled = self._model_autocast_device()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled), torch.no_grad():
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

    def _run_tokenized_repeated_condition(
        self,
        tokenized_sequences: Sequence[TokenizedSequence],
        desc_encoded: dict[str, Any],
    ):
        """Run many DNA rows with one repeated condition using the model shortcut."""

        import torch

        if not tokenized_sequences:
            raise ValueError("tokenized_sequences must be non-empty.")

        input_ids, attention_mask = self._stack_tokenized_dna(tokenized_sequences)
        n = len(tokenized_sequences)
        desc_input_ids = desc_encoded["desc_input_ids"].repeat(n, 1).unsqueeze(0).to(self.device)
        desc_attention_mask = desc_encoded["desc_attention_mask"].repeat(n, 1).unsqueeze(0).to(self.device)
        dataset_flag = torch.zeros((1, n), dtype=torch.bool, device=self.device)

        device_type, autocast_enabled = self._model_autocast_device()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled), torch.no_grad():
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
        desc_encodings: Sequence[dict[str, Any]],
    ):
        """Run one tokenized DNA sequence against many conditions.

        The expression model uses ``dataset_flag=True`` to compute the DNA
        representation once and reuse it for every description in the group.
        This uses the expression model's ``dataset_flag=True`` convention.
        """

        import torch

        if not desc_encodings:
            raise ValueError("desc_encodings must be non-empty.")

        desc_input_ids, desc_attention_mask = self._stack_description_encodings(desc_encodings)
        desc_input_ids = desc_input_ids.unsqueeze(0)
        desc_attention_mask = desc_attention_mask.unsqueeze(0)

        n = len(desc_encodings)
        # The model still expects B*N physical rows; dataset_flag=True only
        # tells it that these DNA rows are duplicates and can be encoded once.
        input_ids = tokenized_sequence.input_ids.unsqueeze(0).repeat(n, 1).to(self.device)
        attention_mask = tokenized_sequence.attention_mask.unsqueeze(0).repeat(n, 1).to(self.device)
        dataset_flag = torch.ones((1, n), dtype=torch.bool, device=self.device)
        device_type, autocast_enabled = self._model_autocast_device()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=autocast_enabled), torch.no_grad():
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

    def _predict_tokenized_sequence(
        self,
        tokenized: TokenizedSequence,
        condition_obj: Condition,
        *,
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        return_tokens: bool = True,
    ) -> ExpressionPrediction:
        """Run one already-tokenized sequence/condition row."""

        prediction = self._predict_many_tokenized_sequences(
            [tokenized],
            [condition_obj],
            grouping=grouping,
            batch_method="predict_sequence",
            return_tokens=return_tokens,
        )[0]
        return self._to_expression_prediction(prediction)

    def predict_sequence(
        self,
        sequence: AnnotatedSequence | str,
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        strand: str = "+",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        return_tokens: bool = True,
    ) -> ExpressionPrediction:
        """Predict expression for one sequence under one condition."""

        condition_obj = self._as_condition(condition)
        tokenized = self.tokenize_sequence(sequence, center=center, strand=strand)
        return self._predict_tokenized_sequence(
            tokenized,
            condition_obj,
            grouping=grouping,
            return_tokens=return_tokens,
        )

    def _predict_many_tokenized_sequences(
        self,
        tokenized_sequences: Sequence[TokenizedSequence],
        conditions: Sequence[Condition],
        *,
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
        batch_method: str,
        return_tokens: bool = True,
        extra_provenance: Mapping[str, Any] | None = None,
    ) -> list[Prediction]:
        """Core row-wise prediction implementation for already-tokenized sequences."""

        tokenized_list = list(tokenized_sequences)
        condition_list = list(conditions)
        if len(tokenized_list) != len(condition_list):
            raise ValueError("tokenized_sequences and conditions must have the same length.")
        if not tokenized_list:
            return []
        valid_grouping = {"no_grouping", "condition", "sequence", "auto"}
        if grouping not in valid_grouping:
            raise ValueError(f"grouping must be one of {sorted(valid_grouping)}, got {grouping!r}.")

        condition_groups: dict[str, list[int]] = {}
        for idx, condition in enumerate(condition_list):
            condition_groups.setdefault(self._grouping_key_for_condition(condition), []).append(idx)

        no_shortcut_condition_groups: dict[tuple[str, str], list[int]] = {}
        for idx, condition in enumerate(condition_list):
            no_shortcut_condition_groups.setdefault(self._forward_grouping_key_for_condition(condition), []).append(idx)

        sequence_groups: dict[tuple[int, ...], list[int]] = {}
        for idx, tokenized in enumerate(tokenized_list):
            sequence_groups.setdefault(self._tokenized_sequence_key(tokenized), []).append(idx)

        desc_encoded_list = self._tokenize_descriptions(condition_list, preprocessing_workers=preprocessing_workers)

        resolved_grouping = grouping
        if grouping == "auto":
            condition_saved = len(tokenized_list) - len(condition_groups)
            sequence_saved = len(tokenized_list) - len(sequence_groups)
            if max(condition_saved, sequence_saved) <= 0:
                resolved_grouping = "no_grouping"
            elif sequence_saved > condition_saved:
                resolved_grouping = "sequence"
            else:
                resolved_grouping = "condition"

        output: list[Prediction | None] = [None] * len(tokenized_list)
        forward_group_id = 0
        extra = dict(extra_provenance or {})

        def store_predictions(
            indices: Sequence[int],
            logits: Any,
            *,
            grouping_used: str,
            group_key_kind: str,
            group_key: str,
            dataset_flag_shape: tuple[int, int],
            dataset_flag_meaning: str,
        ) -> None:
            nonlocal forward_group_id
            for row_pos, idx in enumerate(indices):
                row_logits = logits[row_pos : row_pos + 1]
                expression = self._expression_from_logits(row_logits)
                prediction = Prediction(
                    sequence=tokenized_list[idx].source,
                    condition=condition_list[idx],
                    logits=row_logits,
                    outputs={"expression": expression, "logits": row_logits},
                    tokens=tokenized_list[idx] if return_tokens else None,
                    provenance={
                        **self.provenance,
                        **extra,
                        "batch_method": batch_method,
                        "grouping": grouping,
                        "resolved_grouping": grouping_used,
                        "group_key_kind": group_key_kind,
                        "group_key": group_key,
                        "forward_group_id": forward_group_id,
                        "group_size": len(indices),
                        "dataset_flag_shape": dataset_flag_shape,
                        "dataset_flag_meaning": dataset_flag_meaning,
                        "max_records_per_forward": max_records_per_forward,
                        "record_index": idx,
                    },
                )
                output[idx] = self._attach_description_tokens(prediction, desc_encoded_list[idx])
            forward_group_id += 1

        if resolved_grouping == "no_grouping":
            for condition_key, indices in no_shortcut_condition_groups.items():
                for chunk in self._chunks(indices, max_records_per_forward):
                    model_output = self._run_tokenized_no_grouping(
                        [tokenized_list[idx] for idx in chunk],
                        [desc_encoded_list[idx] for idx in chunk],
                    )
                    store_predictions(
                        chunk,
                        model_output.logits.detach().cpu(),
                        grouping_used="no_grouping",
                        group_key_kind="condition_identity",
                        group_key=f"{condition_key[0]}:{condition_key[1]}",
                        dataset_flag_shape=(len(chunk), 1),
                        dataset_flag_meaning="notebook-compatible layout; B=len(rows for one condition), N=1",
                    )
        elif resolved_grouping == "condition":
            for condition_key, indices in condition_groups.items():
                desc_encoded = desc_encoded_list[indices[0]]
                for chunk in self._chunks(indices, max_records_per_forward):
                    model_output = self._run_tokenized_repeated_condition(
                        [tokenized_list[idx] for idx in chunk],
                        desc_encoded,
                    )
                    store_predictions(
                        chunk,
                        model_output.logits.detach().cpu(),
                        grouping_used="condition",
                        group_key_kind="condition_text",
                        group_key=condition_key,
                        dataset_flag_shape=(1, len(chunk)),
                        dataset_flag_meaning="condition repeated; encode description once and reuse for DNA rows",
                    )
        elif resolved_grouping == "sequence":
            for sequence_key, indices in sequence_groups.items():
                sequence_name = tokenized_list[indices[0]].source.name or f"sequence_index:{indices[0]}"
                for chunk in self._chunks(indices, max_records_per_forward):
                    model_output = self._run_tokenized_same_sequence(
                        tokenized_list[chunk[0]],
                        [desc_encoded_list[idx] for idx in chunk],
                    )
                    store_predictions(
                        chunk,
                        model_output.logits.detach().cpu(),
                        grouping_used="sequence",
                        group_key_kind="tokenized_sequence",
                        group_key=str(sequence_name),
                        dataset_flag_shape=(1, len(chunk)),
                        dataset_flag_meaning="DNA repeated; encode sequence once and reuse for descriptions",
                    )
        else:
            raise RuntimeError(f"Unexpected resolved grouping: {resolved_grouping!r}")

        return [prediction for prediction in output if prediction is not None]

    def predict_multiple_sequences(
        self,
        sequences: AnnotatedSequence | str | Iterable[AnnotatedSequence | str],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        condition: Condition | str | Mapping[str, Any] | None = None,
        center: int | str | Feature = "tss",
        strand: str = "+",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
        return_tokens: bool = True,
    ) -> list[ExpressionPrediction]:
        """Predict expression for sequence/condition rows.

        Accepts many sequences with one condition, one sequence with many
        conditions, or paired sequence/condition rows.
        """

        sequence_list, condition_list = self._sequence_condition_rows(sequences, conditions, condition)
        tokenized = self._tokenize_sequence_tasks(
            [(sequence, center, strand) for sequence in sequence_list],
            preprocessing_workers=preprocessing_workers,
        )
        predictions = self._predict_many_tokenized_sequences(
            tokenized,
            condition_list,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
            batch_method="predict_multiple_sequences",
            return_tokens=return_tokens,
        )
        return [self._to_expression_prediction(prediction) for prediction in predictions]

    def _predict_multiple_sequences_one_condition(
        self,
        sequences: Iterable[AnnotatedSequence | str],
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        strand: str = "+",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
        return_tokens: bool = True,
    ) -> list[ExpressionPrediction]:
        """Predict many sequences under one condition."""

        return self.predict_multiple_sequences(
            sequences,
            condition=condition,
            center=center,
            strand=strand,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
            return_tokens=return_tokens,
        )

    def _predict_one_sequence_many_conditions(
        self,
        sequence: AnnotatedSequence | str,
        conditions: Iterable[Condition | str | Mapping[str, Any]],
        *,
        center: int | str | Feature = "tss",
        strand: str = "+",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
        return_tokens: bool = True,
    ) -> list[ExpressionPrediction]:
        """Predict one sequence under many conditions."""

        return self.predict_multiple_sequences(
            sequence,
            conditions=conditions,
            center=center,
            strand=strand,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
            return_tokens=return_tokens,
        )

    def predict_pair(
        self,
        pair: SequencePair,
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
    ) -> PairPrediction:
        """Predict reference and alternative sequences for a sequence pair."""

        return self.predict_multiple_pairs(
            [pair],
            condition=condition,
            center=center,
            grouping=grouping,
        )[0]

    def _predict_multiple_pairs_one_condition(
        self,
        pairs: Iterable[SequencePair],
        *,
        condition: Condition | str | Mapping[str, Any],
        center: int | str | Feature = "tss",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
    ) -> list[PairPrediction]:
        """Predict many reference/alternative pairs under one condition."""

        return self.predict_multiple_pairs(
            pairs,
            condition=condition,
            center=center,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
        )

    def predict_multiple_pairs(
        self,
        pairs: Iterable[SequencePair],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        condition: Condition | str | Mapping[str, Any] | None = None,
        center: int | str | Feature = "tss",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
    ) -> list[PairPrediction]:
        """Predict row-wise sequence-pair/condition batches."""

        pair_list = list(pairs)
        condition_list = self._broadcast_conditions(len(pair_list), conditions, condition)
        if not pair_list:
            return []

        centers = [self._pair_center_or_variant(pair, center) for pair in pair_list]
        ref_tokenized = self._tokenize_sequence_tasks(
            [(pair.ref, pair_center, "+") for pair, pair_center in zip(pair_list, centers)],
            preprocessing_workers=preprocessing_workers,
        )
        alt_tokenized = self._tokenize_sequence_tasks(
            [(pair.alt, pair_center, "+") for pair, pair_center in zip(pair_list, centers)],
            preprocessing_workers=preprocessing_workers,
        )

        ref_predictions = self._predict_many_tokenized_sequences(
            ref_tokenized,
            condition_list,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
            batch_method="many_pair_refs",
            return_tokens=True,
        )
        alt_predictions = self._predict_many_tokenized_sequences(
            alt_tokenized,
            condition_list,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
            batch_method="many_pair_alts",
            return_tokens=True,
        )
        return [
            PairPrediction(
                ref=self._to_expression_prediction(ref_pred),
                alt=self._to_expression_prediction(alt_pred),
                pair=pair,
                condition=condition,
            )
            for pair, condition, ref_pred, alt_pred in zip(pair_list, condition_list, ref_predictions, alt_predictions)
        ]

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

    def predict_tss(
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
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
        return_sequence: bool = True,
    ) -> ExpressionPrediction:
        """Predict expression directly from a genome and TSS coordinate."""

        tss0 = int(tss) - 1 if coordinate_system == "1-based" else int(tss)
        record = {"chrom": chrom, "tss": tss, "strand": strand, "name": f"{chrom}:{tss0}:{strand}"}
        return self.predict_multiple_tss(
            [record],
            condition=condition,
            genome=genome,
            context_bp=context_bp,
            coordinate_system=coordinate_system,
            include_features=include_features,
            center_feature_name=center_feature_name,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
            return_sequence=return_sequence,
        )[0]

    @staticmethod
    def _record_field(record: Any, field: str, default: Any = None) -> Any:
        """Return ``field`` from a mapping or row-like object."""

        if isinstance(record, Mapping):
            return record.get(field, default)
        return getattr(record, field, default)

    @staticmethod
    def _chunks(indices: Sequence[int], size: int | None):
        if size is None:
            yield list(indices)
            return
        if size <= 0:
            raise ValueError("max_records_per_forward must be positive.")
        for start in range(0, len(indices), size):
            yield list(indices[start : start + size])

    @staticmethod
    def _tokenized_sequence_key(tokenized: TokenizedSequence) -> tuple[int, ...]:
        return tuple(int(value) for value in tokenized.input_ids.tolist())

    @staticmethod
    def _grouping_key_for_condition(condition: Condition) -> str:
        return condition.text()

    @staticmethod
    def _forward_grouping_key_for_condition(condition: Condition) -> tuple[str, str]:
        return condition.name, condition.text()

    @staticmethod
    def _worker_genome_config(genome: Any) -> dict[str, Any]:
        if not hasattr(genome, "fasta_path"):
            raise TypeError(
                "preprocessing_workers requires a Genome-like object with fasta_path, "
                "annotation, build, and chrom_style attributes."
            )
        annotation = getattr(genome, "annotation", None)
        if isinstance(annotation, Path):
            annotation = str(annotation)
        return {
            "fasta_path": str(getattr(genome, "fasta_path")),
            "annotation": annotation,
            "build": getattr(genome, "build", None),
            "chrom_style": getattr(genome, "chrom_style", "auto"),
            "cache": True,
        }

    def _worker_tokenizer_config(self) -> dict[str, Any]:
        name_or_path = getattr(self.dna_tokenizer, "name_or_path", None)
        if not name_or_path:
            raise ValueError(
                "preprocessing_workers requires dna_tokenizer.name_or_path so each worker can load its own tokenizer."
            )
        return {
            "name_or_path": str(name_or_path),
            "dna_max_seq_len": self.dna_max_seq_len,
            "token_len_for_fetch": self.token_len_for_fetch,
            "num_before": self.num_before,
            "cls_id": self.dna_tokenizer.cls_token_id,
            "sep_id": self.dna_tokenizer.sep_token_id,
            "pad_id": self.dna_tokenizer.pad_token_id,
        }

    def _prepare_tss_tokenized_records(
        self,
        records: Sequence[Any],
        *,
        genome: Any,
        context_bp: int | None,
        coordinate_system: Literal["0-based", "1-based"],
        include_features: bool,
        center_feature_name: str,
        preprocessing_workers: int,
    ) -> list[TokenizedSequence]:
        """Build and tokenize TSS-centered sequences, optionally in worker processes."""

        tasks = []
        for idx, record in enumerate(records):
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
            tasks.append(
                {
                    "index": idx,
                    "chrom": str(chrom),
                    "tss0": tss0,
                    "strand": strand,
                    "name": name or f"{chrom}:{tss0}:{strand}",
                    "size": size,
                    "include_features": include_features,
                    "center_feature_name": center_feature_name,
                }
            )

        output: list[TokenizedSequence | None] = [None] * len(tasks)
        if preprocessing_workers <= 0:
            for task in tasks:
                sequence = genome.sequence_around(
                    task["chrom"],
                    task["tss0"],
                    task["size"],
                    strand=task["strand"],
                    include_features=task["include_features"],
                    center_feature_name=task["center_feature_name"],
                    name=task["name"],
                )
                output[task["index"]] = self.tokenize_sequence(sequence, center=center_feature_name, strand="+")
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            chunksize = max(1, len(tasks) // (int(preprocessing_workers) * 4)) if tasks else 1
            with ctx.Pool(
                processes=int(preprocessing_workers),
                initializer=_tss_worker_init,
                initargs=(self._worker_genome_config(genome), self._worker_tokenizer_config()),
            ) as pool:
                for idx, tokenized in pool.imap_unordered(_tss_worker_tokenize, tasks, chunksize=chunksize):
                    output[idx] = tokenized

        return [tokenized for tokenized in output if tokenized is not None]

    def predict_multiple_tss(
        self,
        records: Iterable[Any],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        condition: Condition | str | Mapping[str, Any] | None = None,
        genome: Any,
        context_bp: int | None = None,
        coordinate_system: Literal["0-based", "1-based"] = "0-based",
        include_features: bool = True,
        center_feature_name: str = "tss",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
        return_sequence: bool = True,
    ) -> list[ExpressionPrediction]:
        """Predict expression for paired TSS records and conditions.

        ``records`` can be paired with one condition or row-wise conditions.
        TSS sequence building uses process workers when ``preprocessing_workers``
        is positive; each worker opens its own genome handle.
        """

        record_list = list(records)
        condition_list = self._broadcast_conditions(len(record_list), conditions, condition)
        if not record_list:
            return []

        tokenized_list = self._prepare_tss_tokenized_records(
            record_list,
            genome=genome,
            context_bp=context_bp,
            coordinate_system=coordinate_system,
            include_features=include_features,
            center_feature_name=center_feature_name,
            preprocessing_workers=preprocessing_workers,
        )
        if len(tokenized_list) != len(record_list):
            raise RuntimeError("TSS preprocessing did not return one tokenized sequence per record.")

        predictions = self._predict_many_tokenized_sequences(
            tokenized_list,
            condition_list,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
            batch_method="predict_multiple_tss",
            return_tokens=True,
            extra_provenance={"preprocessing_workers": preprocessing_workers},
        )
        return [self._to_expression_prediction(prediction, return_sequence=return_sequence) for prediction in predictions]


class VariantInterpreter:
    """High-level object for scoring variant and sequence-pair effects."""

    def __init__(self, model: SequenceModel) -> None:
        self.model = model

    def score_variant(
        self,
        variant: Any,
        *,
        context: Any | None = None,
        condition: Condition | str | Mapping[str, Any],
        scorer: Any,
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
        **sequence_pair_kwargs: Any,
    ) -> ScoringResult | VariantReport:
        """Build one sequence pair with :meth:`Variant.to_sequence_pair` and score it."""

        pair, label = self._pair_from_variant_record(
            variant,
            context=context,
            genome=genome,
            coordinate_system=coordinate_system,
            fallback="variant",
            **sequence_pair_kwargs,
        )
        return self.score_sequence_pair(
            pair,
            condition=condition,
            scorer=scorer,
            center=center,
            grouping=grouping,
            label=label,
        )

    def score_variants(
        self,
        variants: Iterable[Any],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        context: Any | None = None,
        condition: Condition | str | Mapping[str, Any] | None = None,
        scorer: Any,
        genome: Any | None = None,
        center: int | str | Feature = "tss",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
        coordinate_system: Literal["auto", "0-based", "1-based"] = "auto",
        on_error: Literal["raise", "warn", "skip", "exit"] = "raise",
        **sequence_pair_kwargs: Any,
    ) -> list[ScoringResult | VariantReport]:
        """Score variant records with one broadcast condition or row-wise conditions."""

        record_list = list(variants)
        if not record_list:
            return []
        condition_list = self.model._broadcast_conditions(len(record_list), conditions, condition)
        self._validate_on_error(on_error)

        if on_error in {"warn", "skip"}:
            results: list[ScoringResult | VariantReport] = []
            for idx, (record, row_condition) in enumerate(zip(record_list, condition_list)):
                try:
                    pair, label = self._pair_from_variant_record(
                        record,
                        context=context,
                        genome=genome,
                        coordinate_system=coordinate_system,
                        fallback=f"variant_{idx}",
                        **sequence_pair_kwargs,
                    )
                    results.extend(
                        self._score_pairs(
                            [pair],
                            [row_condition],
                            [label],
                            scorer=scorer,
                            center=center,
                            grouping=grouping,
                            preprocessing_workers=preprocessing_workers,
                            max_records_per_forward=max_records_per_forward,
                        )
                    )
                except Exception as exc:
                    self._handle_scoring_error(exc, item=record, on_error=on_error)
            return results

        try:
            pairs: list[SequencePair] = []
            labels: list[str] = []
            for idx, record in enumerate(record_list):
                pair, label = self._pair_from_variant_record(
                    record,
                    context=context,
                    genome=genome,
                    coordinate_system=coordinate_system,
                    fallback=f"variant_{idx}",
                    **sequence_pair_kwargs,
                )
                pairs.append(pair)
                labels.append(label)
            return self._score_pairs(
                pairs,
                condition_list,
                labels,
                scorer=scorer,
                center=center,
                grouping=grouping,
                preprocessing_workers=preprocessing_workers,
                max_records_per_forward=max_records_per_forward,
            )
        except Exception as exc:
            self._handle_scoring_error(exc, item="variant batch", on_error=on_error)
            return []

    def score_sequence_pair(
        self,
        pair: SequencePair,
        *,
        condition: Condition | str | Mapping[str, Any],
        scorer: Any,
        center: int | str | Feature = "tss",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        label: str | None = None,
    ) -> ScoringResult | VariantReport:
        """Predict and score an already materialized sequence pair."""

        return self._score_pairs(
            [pair],
            [self.model._as_condition(condition)],
            [label or self._pair_label(pair, 0)],
            scorer=scorer,
            center=center,
            grouping=grouping,
        )[0]

    def score_sequence_pairs(
        self,
        pairs: Iterable[SequencePair],
        conditions: Iterable[Condition | str | Mapping[str, Any]] | None = None,
        *,
        condition: Condition | str | Mapping[str, Any] | None = None,
        scorer: Any,
        center: int | str | Feature = "tss",
        grouping: Literal["no_grouping", "condition", "sequence", "auto"] = "no_grouping",
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
        on_error: Literal["raise", "warn", "skip", "exit"] = "raise",
    ) -> list[ScoringResult | VariantReport]:
        """Score sequence pairs with one broadcast condition or row-wise conditions."""

        pair_list = list(pairs)
        if not pair_list:
            return []
        condition_list = self.model._broadcast_conditions(len(pair_list), conditions, condition)
        labels = [self._pair_label(pair, idx) for idx, pair in enumerate(pair_list)]
        self._validate_on_error(on_error)

        if on_error in {"warn", "skip"}:
            results: list[ScoringResult | VariantReport] = []
            for pair, row_condition, label in zip(pair_list, condition_list, labels):
                try:
                    results.extend(
                        self._score_pairs(
                            [pair],
                            [row_condition],
                            [label],
                            scorer=scorer,
                            center=center,
                            grouping=grouping,
                            preprocessing_workers=preprocessing_workers,
                            max_records_per_forward=max_records_per_forward,
                        )
                    )
                except Exception as exc:
                    self._handle_scoring_error(exc, item=label, on_error=on_error)
            return results

        try:
            return self._score_pairs(
                pair_list,
                condition_list,
                labels,
                scorer=scorer,
                center=center,
                grouping=grouping,
                preprocessing_workers=preprocessing_workers,
                max_records_per_forward=max_records_per_forward,
            )
        except Exception as exc:
            self._handle_scoring_error(exc, item="sequence-pair batch", on_error=on_error)
            return []

    def _score_pairs(
        self,
        pairs: Sequence[SequencePair],
        conditions: Sequence[Condition],
        labels: Sequence[str],
        *,
        scorer: Any,
        center: int | str | Feature,
        grouping: Literal["no_grouping", "condition", "sequence", "auto"],
        preprocessing_workers: int = 0,
        max_records_per_forward: int | None = None,
    ) -> list[ScoringResult | VariantReport]:
        predictions = self.model.predict_multiple_pairs(
            pairs,
            conditions=conditions,
            center=center,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            max_records_per_forward=max_records_per_forward,
        )
        return [
            self._name_scoring_output(scorer.score(prediction), label=label)
            for prediction, label in zip(predictions, labels)
        ]

    def _pair_from_variant_record(
        self,
        record: Any,
        *,
        context: Any | None,
        genome: Any | None,
        coordinate_system: Literal["auto", "0-based", "1-based"],
        fallback: str,
        **sequence_pair_kwargs: Any,
    ) -> tuple[SequencePair, str]:
        variant = self._variant_from_record(record, genome=genome, coordinate_system=coordinate_system)
        explicit_name = sequence_pair_kwargs.pop("name", None)
        label = str(explicit_name or self._variant_record_name(record, variant=variant, fallback=fallback))
        pair = variant.to_sequence_pair(
            context=context,
            genome=genome,
            name=label,
            **sequence_pair_kwargs,
        )
        return pair, label

    @staticmethod
    def _variant_record_name(record: Any, *, variant: Variant, fallback: str) -> str:
        """Return a stable label for a variant record."""

        if variant.id:
            return str(variant.id)
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

    @staticmethod
    def _pair_label(pair: SequencePair, index: int) -> str:
        variant = pair.variant
        if hasattr(variant, "id") and variant.id:
            return str(variant.id)
        metadata = dict(pair.metadata or {})
        for key in ("name", "id", "variant_id"):
            if metadata.get(key):
                return str(metadata[key])
        if pair.ref.name and pair.ref.name == pair.alt.name:
            return pair.ref.name
        return f"pair_{index}"

    @staticmethod
    def _name_scoring_output(output: ScoringResult | VariantReport, *, label: str) -> ScoringResult | VariantReport:
        if isinstance(output, ScoringResult):
            output.name = f"{label}:{output.name}"
            return output
        if isinstance(output, VariantReport):
            for key, result in output.results.items():
                scorer_name = result.name or key
                result.name = f"{label}:{scorer_name}"
            return output
        raise TypeError("scorer.score(...) must return ScoringResult or VariantReport.")

    @staticmethod
    def _validate_on_error(on_error: str) -> None:
        if on_error not in {"raise", "warn", "skip", "exit"}:
            raise ValueError("on_error must be 'raise', 'warn', 'skip', or 'exit'.")

    @staticmethod
    def _handle_scoring_error(exc: Exception, *, item: Any, on_error: str) -> None:
        if on_error == "raise":
            raise exc
        if on_error == "exit":
            raise SystemExit(f"Scoring failed for {item}: {exc}") from exc
        if on_error == "warn":
            import warnings

            warnings.warn(f"Skipping {item}: {exc}", RuntimeWarning, stacklevel=2)
