"""Model loading, preprocessing orchestration, and sequence prediction."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..conditions import Condition, metadata_to_description
from ..config import (
    FULL_RETENTION,
    PredictionRetention,
    RetentionLike,
    retention_policy_for,
)
from ..sequences import AnnotatedSequence, Feature, SequencePair
from .batching import (
    GroupingMode,
    PairExecutionMode,
    PreprocessingBackend,
    _ForwardBatch,
    _process_worker_init,
    _process_worker_tokenize_description,
    _process_worker_tokenize_sequence,
    _progress,
    _thread_worker_init,
)
from .outputs import ExpressionPrediction, PairPrediction, Prediction, _simple_value
from .tokenization import CenteredTokenizer, TokenizedSequence


def _split_model_class_spec(model_cls: str) -> tuple[str, str]:
    """Accept the repo-standard ``module:Class`` and API ``file::Class`` forms."""

    separator = "::" if "::" in model_cls else ":"
    parts = model_cls.rsplit(separator, 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "model_cls must use 'module:ClassName' or 'path/to/file.py::ClassName', "
            f"got {model_cls!r}"
        )
    return parts[0], parts[1]


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
        """Attach model runtime objects and configure centered tokenization."""

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

        ``model_cls`` accepts the repository-standard ``package.module:ClassName``
        or the API's explicit ``/path/to/file.py::ClassName`` form.
        """

        import torch
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        logger = logging.getLogger(__name__)
        model_path_or_module, class_name = _split_model_class_spec(model_cls)

        @contextmanager
        def _temporary_sys_path(path: Path):
            """Temporarily prepend a model directory to ``sys.path``."""

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
            """Load the configured model class from a file or module."""

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
        """Normalize user condition input to a :class:`Condition`."""

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
        """Return the autocast device type and whether it should be enabled."""

        device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
        return device_type, device_type == "cuda"

    @staticmethod
    def _validate_preprocessing_backend(backend: PreprocessingBackend) -> None:
        """Validate a CPU preprocessing backend name."""

        if backend not in {"thread", "process"}:
            raise ValueError("preprocessing_backend must be 'thread' or 'process'.")

    @contextmanager
    def _preprocessing_executor(
        self,
        backend: PreprocessingBackend,
        workers: int,
    ):
        """Yield one executor shared by every preprocessing stage in an API call."""

        if workers < 0:
            raise ValueError("preprocessing_workers must be non-negative.")
        self._validate_preprocessing_backend(backend)
        if workers == 0:
            yield None
            return

        self.logger.info("Starting %s preprocessing with %d worker(s)", backend, workers)
        if backend == "thread":
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(
                max_workers=int(workers),
                initializer=_thread_worker_init,
                initargs=(self.centered_tokenizer,),
            ) as executor:
                yield executor
            return

        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=mp.get_context("spawn"),
            initializer=_process_worker_init,
            initargs=(
                self._worker_tokenizer_config(),
                self._worker_description_tokenizer_config(),
            ),
        ) as executor:
            yield executor

    @staticmethod
    def _map_preprocessing(
        func: Any,
        items: Sequence[Any],
        preprocessing_workers: int,
        *,
        executor: Any | None = None,
        show_progress: bool = False,
        progress_description: str = "Preprocessing",
    ) -> list[Any]:
        """Map CPU preprocessing while preserving input order."""

        if preprocessing_workers < 0:
            raise ValueError("preprocessing_workers must be non-negative.")
        item_list = list(items)
        if preprocessing_workers == 0 or len(item_list) < 2:
            iterator = _progress(
                item_list,
                total=len(item_list),
                description=progress_description,
                enabled=show_progress,
                unit="item",
            )
            return [func(item) for item in iterator]

        if executor is None:
            raise RuntimeError("Parallel preprocessing requires an initialized executor.")

        from concurrent.futures import as_completed

        futures = {executor.submit(func, item): idx for idx, item in enumerate(item_list)}
        output: list[Any] = [None] * len(item_list)
        completed = _progress(
            as_completed(futures),
            total=len(futures),
            description=progress_description,
            enabled=show_progress,
            unit="item",
        )
        for future in completed:
            output[futures[future]] = future.result()
        return output

    @staticmethod
    def _as_sequence_list(sequences: AnnotatedSequence | str | Iterable[AnnotatedSequence | str]) -> list[AnnotatedSequence | str]:
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

    @staticmethod
    def _sequence_task_key(sequence: AnnotatedSequence | str, center: int | str | Feature, strand: str) -> tuple[Any, ...]:
        """Return a stable in-process cache key for one tokenization task."""

        sequence_key = ("str", sequence) if isinstance(sequence, str) else ("object", id(sequence))
        center_key = ("value", center) if isinstance(center, (int, str)) else ("object", id(center))
        return sequence_key, center_key, strand

    def _sequence_process_payload(
        self,
        task: tuple[AnnotatedSequence | str, int | str | Feature, str],
    ) -> tuple[dict[str, Any], AnnotatedSequence]:
        """Return a compact process task plus the original parent-side source."""

        sequence, center, strand = task
        source = self.centered_tokenizer._as_annotated(sequence)
        center_index = self.centered_tokenizer._resolve_center(source, center)
        features = tuple(
            {
                "name": feature.name,
                "start": feature.start,
                "end": feature.end,
                "type": feature.type,
                "strand": feature.strand,
                "source": feature.source,
            }
            for feature in source.features
        )
        return (
            {
                "sequence": source.sequence,
                "name": source.name,
                "features": features,
                "center": center_index,
                "strand": strand,
            },
            source,
        )

    @staticmethod
    def _restore_process_tokenized_sequence(
        payload: Mapping[str, Any],
        source: AnnotatedSequence,
    ) -> TokenizedSequence:
        """Reattach parent-side sequence metadata to compact worker output."""

        import torch

        return TokenizedSequence(
            input_ids=torch.tensor(payload["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(payload["attention_mask"], dtype=torch.long),
            tokens=list(payload["tokens"]),
            source=source,
            center=int(payload["center"]),
            strand=str(payload["strand"]),
        )

    @staticmethod
    def _restore_process_description(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a compact description worker result to the model contract."""

        import torch

        return {
            "description": payload["description"],
            "desc_input_ids": torch.tensor(payload["desc_input_ids"], dtype=torch.long),
            "desc_attention_mask": torch.tensor(payload["desc_attention_mask"], dtype=torch.long),
        }

    def _tokenize_sequence_tasks(
        self,
        tasks: Sequence[tuple[AnnotatedSequence | str, int | str | Feature, str]],
        *,
        preprocessing_workers: int = 0,
        preprocessing_backend: PreprocessingBackend = "process",
        preprocessing_executor: Any | None = None,
        show_progress: bool = False,
        progress_description: str = "Tokenizing DNA",
    ) -> list[TokenizedSequence]:
        """Tokenize unique DNA tasks and restore the original row order."""

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
            """Tokenize one normalized sequence task."""

            sequence, center, strand = task
            return self.tokenize_sequence(sequence, center=center, strand=strand)

        self.logger.info("%s: %d unique sequence(s)", progress_description, len(unique_tasks))
        if preprocessing_backend == "process" and preprocessing_workers > 0 and len(unique_tasks) >= 2:
            process_tasks = [self._sequence_process_payload(task) for task in unique_tasks]
            raw_payloads = self._map_preprocessing(
                _process_worker_tokenize_sequence,
                [payload for payload, _ in process_tasks],
                preprocessing_workers,
                executor=preprocessing_executor,
                show_progress=show_progress,
                progress_description=progress_description,
            )
            tokenized_unique = [
                self._restore_process_tokenized_sequence(payload, source)
                for payload, (_, source) in zip(raw_payloads, process_tasks)
            ]
        else:
            tokenized_unique = self._map_preprocessing(
                tokenize,
                unique_tasks,
                preprocessing_workers if preprocessing_backend == "thread" else 0,
                executor=preprocessing_executor,
                show_progress=show_progress,
                progress_description=progress_description,
            )
        tokenized_by_key = dict(zip(unique_keys, tokenized_unique))
        return [tokenized_by_key[key] for key in row_keys]

    @staticmethod
    def _description_token_payload(desc_encoded: Mapping[str, Any]) -> dict[str, Any]:
        """Expose description tensors under compatibility field names."""

        return {
            "description": desc_encoded.get("description"),
            "input_ids": desc_encoded["desc_input_ids"],
            "attention_mask": desc_encoded["desc_attention_mask"],
            "desc_input_ids": desc_encoded["desc_input_ids"],
            "desc_attention_mask": desc_encoded["desc_attention_mask"],
        }

    def _attach_description_tokens(self, prediction: Prediction, desc_encoded: Mapping[str, Any]) -> Prediction:
        """Attach reusable description-token metadata to a prediction."""

        payload = self._description_token_payload(desc_encoded)
        prediction.description_tokens = payload
        return prediction

    @staticmethod
    def _copy_description_tokens(source: Prediction, target: Prediction) -> Prediction:
        """Copy description-token metadata between prediction containers."""

        target.description_tokens = source.description_tokens
        return target

    def _to_expression_prediction(self, prediction: Prediction, *, return_sequence: bool = True) -> ExpressionPrediction:
        """Wrap a raw prediction with its derived expression value."""

        expression_prediction = ExpressionPrediction(
            sequence=prediction.sequence if return_sequence else None,
            sequence_name=prediction.sequence_name,
            condition=prediction.condition,
            logits=prediction.logits,
            outputs=prediction.outputs,
            tokens=prediction.tokens,
            provenance=prediction.provenance,
            description_tokens=prediction.description_tokens,
            expression=prediction.scalar("expression"),
        )
        return self._copy_description_tokens(prediction, expression_prediction)

    def _run_tokenized_no_grouping(
        self,
        tokenized_sequences: Sequence[TokenizedSequence],
        desc_encodings: Sequence[dict[str, Any]],
    ):
        """Run batched row-wise pairs with ``B=len(rows), N=1`` and no shortcut."""

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

    def _run_tokenized_pairs(
        self,
        tokenized_sequences: Sequence[TokenizedSequence],
        desc_encodings: Sequence[dict[str, Any]],
    ):
        """Run interleaved reference/alternative rows in pair-aware blocks."""

        import torch

        if not tokenized_sequences or len(tokenized_sequences) % 2:
            raise ValueError("tokenized_sequences must contain complete reference/alternative pairs.")
        if len(tokenized_sequences) != len(desc_encodings):
            raise ValueError("tokenized_sequences and desc_encodings must have the same length.")

        input_ids, attention_mask = self._stack_tokenized_dna(tokenized_sequences)
        pair_descriptions = desc_encodings[::2]
        desc_input_ids, desc_attention_mask = self._stack_description_encodings(pair_descriptions)

        # B is the number of variants and N=2 is (reference, alternative).
        # The zero flag tells the model that DNA differs inside each block while
        # the condition is repeated and should be encoded once for both alleles.
        desc_input_ids = desc_input_ids.unsqueeze(1).repeat(1, 2, 1)
        desc_attention_mask = desc_attention_mask.unsqueeze(1).repeat(1, 2, 1)
        dataset_flag = torch.zeros((len(pair_descriptions), 2), dtype=torch.bool, device=self.device)

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

    def _run_tokenized_repeated_sequence(
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
    def _logits_to_cpu(model_output: Any) -> Any:
        """Detach model logits and move them off CUDA as soon as possible."""

        return model_output.logits.detach().cpu()

    @staticmethod
    def clear_cuda_cache() -> None:
        """Release unused CUDA cache after model outputs have been copied to CPU."""

        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def _expression_from_logits(logits: Any) -> Any:
        """Extract expression output using the attached first-token convention."""

        return logits[:, 0:1, :].squeeze(-1)

    def _predict_tokenized_sequence(
        self,
        tokenized: TokenizedSequence,
        condition_obj: Condition,
        *,
        grouping: GroupingMode = "no_grouping",
        return_tokens: bool = True,
        retention: PredictionRetention = FULL_RETENTION.prediction,
        description_cache: dict[str, dict[str, Any]] | None = None,
    ) -> ExpressionPrediction:
        """Run one already-tokenized sequence/condition row."""

        prediction = self._predict_multiple_tokenized_sequences(
            [tokenized],
            [condition_obj],
            grouping=grouping,
            prefetch_batches=0,
            show_progress=False,
            batch_method="predict_sequence",
            return_tokens=return_tokens,
            retention=retention,
            description_cache=description_cache,
        )[0]
        return self._to_expression_prediction(prediction)

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
        description_cache: dict[str, dict[str, Any]] | None = None,
    ) -> ExpressionPrediction:
        """Predict expression for one sequence under one condition."""

        condition_obj = self._as_condition(condition)
        prediction_retention = retention_policy_for(retention).prediction
        tokenized = self.tokenize_sequence(sequence, center=center, strand=strand)
        return self._predict_tokenized_sequence(
            tokenized,
            condition_obj,
            grouping=grouping,
            return_tokens=return_tokens,
            retention=prediction_retention,
            description_cache=description_cache,
        )

    def _predict_sequence_tasks(
        self,
        tasks: Sequence[tuple[AnnotatedSequence | str, int | str | Feature, str]],
        conditions: Sequence[Condition],
        *,
        grouping: GroupingMode,
        preprocessing_workers: int,
        preprocessing_backend: PreprocessingBackend,
        preprocessing_executor: Any | None,
        max_records_per_forward: int | None,
        prefetch_batches: int,
        show_progress: bool,
        batch_method: str,
        progress_description: str,
        return_tokens: bool = True,
        retention: PredictionRetention = FULL_RETENTION.prediction,
        description_cache: dict[str, dict[str, Any]] | None = None,
        paired_rows: bool = False,
        max_pairs_per_forward: int | None = None,
    ) -> list[Prediction]:
        """Predict raw sequence tasks, optionally preserving adjacent allele pairs."""

        task_list = list(tasks)
        sequence_tasks = None
        if grouping in {"sequence", "auto"} and not paired_rows:
            dna_stage = {
                "multiple_pair_refs": "Tokenizing references",
                "multiple_pair_alts": "Tokenizing alternatives",
            }.get(batch_method, "Tokenizing DNA")
            tokenized: list[TokenizedSequence | None] = list(
                self._tokenize_sequence_tasks(
                    task_list,
                    preprocessing_workers=preprocessing_workers,
                    preprocessing_backend=preprocessing_backend,
                    preprocessing_executor=preprocessing_executor,
                    show_progress=show_progress,
                    progress_description=dna_stage,
                )
            )
        else:
            # These modes can plan batches without seeing DNA token IDs first.
            tokenized = [None] * len(task_list)
            sequence_tasks = task_list

        return self._predict_multiple_tokenized_sequences(
            tokenized,
            conditions,
            grouping=grouping,
            preprocessing_workers=preprocessing_workers,
            preprocessing_backend=preprocessing_backend,
            preprocessing_executor=preprocessing_executor,
            max_records_per_forward=max_records_per_forward,
            prefetch_batches=prefetch_batches,
            show_progress=show_progress,
            batch_method=batch_method,
            progress_description=progress_description,
            return_tokens=return_tokens,
            description_cache=description_cache,
            sequence_tasks=sequence_tasks,
            paired_rows=paired_rows,
            max_pairs_per_forward=max_pairs_per_forward,
            retention=retention,
        )

    def _predict_multiple_tokenized_sequences(
        self,
        tokenized_sequences: Sequence[TokenizedSequence | None],
        conditions: Sequence[Condition],
        *,
        grouping: GroupingMode = "no_grouping",
        preprocessing_workers: int = 0,
        preprocessing_backend: PreprocessingBackend = "process",
        preprocessing_executor: Any | None = None,
        max_records_per_forward: int | None = None,
        prefetch_batches: int = 1,
        show_progress: bool = True,
        batch_method: str,
        progress_description: str | None = None,
        return_tokens: bool = True,
        extra_provenance: Mapping[str, Any] | None = None,
        description_cache: dict[str, dict[str, Any]] | None = None,
        sequence_tasks: Sequence[tuple[AnnotatedSequence | str, int | str | Feature, str]] | None = None,
        paired_rows: bool = False,
        max_pairs_per_forward: int | None = None,
        retention: PredictionRetention = FULL_RETENTION.prediction,
    ) -> list[Prediction]:
        """Core row-wise prediction implementation for already-tokenized sequences.

        ``serial`` runs one independent pair per forward. ``no_grouping`` keeps
        the same no-shortcut model semantics but batches independent pairs.
        When ``paired_rows`` is true, adjacent rows are treated as one
        reference/alternative pair and can never cross a forward boundary.
        DNA and description tokenization can be prefetched in threads or spawned
        processes while the GPU runs the current batch. ``prefetch_batches=0``
        disables look-ahead.
        """

        from collections import deque
        from concurrent.futures import Future

        tokenized_list = list(tokenized_sequences)
        condition_list = list(conditions)
        if len(tokenized_list) != len(condition_list):
            raise ValueError("tokenized_sequences and conditions must have the same length.")
        task_list = list(sequence_tasks) if sequence_tasks is not None else None
        if task_list is not None and len(task_list) != len(condition_list):
            raise ValueError("sequence_tasks and conditions must have the same length.")
        if task_list is None and any(tokenized is None for tokenized in tokenized_list):
            raise ValueError("Missing tokenized sequences without sequence_tasks to prepare them.")
        if not tokenized_list:
            return []
        if preprocessing_workers < 0:
            raise ValueError("preprocessing_workers must be non-negative.")
        self._validate_preprocessing_backend(preprocessing_backend)
        if preprocessing_workers > 0 and preprocessing_executor is None:
            raise RuntimeError("Parallel preprocessing requires an initialized executor.")
        if prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative.")
        if max_pairs_per_forward is not None and max_pairs_per_forward <= 0:
            raise ValueError("max_pairs_per_forward must be positive.")
        if paired_rows and len(tokenized_list) % 2:
            raise ValueError("paired_rows requires an even number of sequence rows.")
        if paired_rows and max_records_per_forward is not None:
            raise ValueError("Use max_pairs_per_forward for joint pair execution.")
        if not paired_rows and max_pairs_per_forward is not None:
            raise ValueError("max_pairs_per_forward is only valid for joint pair execution.")
        valid_grouping = {"serial", "no_grouping", "condition", "sequence", "auto"}
        if grouping not in valid_grouping:
            raise ValueError(f"grouping must be one of {sorted(valid_grouping)}, got {grouping!r}.")
        if paired_rows and grouping == "sequence":
            raise ValueError(
                "Joint pair execution does not support grouping='sequence'; "
                "use 'no_grouping', 'condition', 'auto', or pair_execution='separate'."
            )

        pair_condition_groups: dict[str, list[int]] = {}
        if paired_rows:
            for pair_idx in range(len(condition_list) // 2):
                ref_idx = 2 * pair_idx
                alt_idx = ref_idx + 1
                ref_key = self._grouping_key_for_condition(condition_list[ref_idx])
                alt_key = self._grouping_key_for_condition(condition_list[alt_idx])
                if ref_key != alt_key:
                    raise ValueError("Reference and alternative rows in a pair must use the same condition.")
                pair_condition_groups.setdefault(ref_key, []).append(pair_idx)

        condition_groups: dict[str, list[int]] = {}
        for idx, condition in enumerate(condition_list):
            condition_groups.setdefault(self._grouping_key_for_condition(condition), []).append(idx)

        sequence_groups: dict[tuple[int, ...], list[int]] = {}
        if grouping in {"sequence", "auto"} and not paired_rows:
            for idx, tokenized in enumerate(tokenized_list):
                if tokenized is None:
                    raise RuntimeError("sequence and auto grouping require tokenized DNA before batch planning.")
                sequence_groups.setdefault(self._tokenized_sequence_key(tokenized), []).append(idx)

        resolved_grouping = grouping
        if paired_rows:
            if grouping == "condition":
                resolved_grouping = "condition"
            elif grouping == "auto" and any(len(indices) > 1 for indices in pair_condition_groups.values()):
                resolved_grouping = "condition"
            else:
                resolved_grouping = "pair"
        elif grouping == "auto":
            condition_saved = len(tokenized_list) - len(condition_groups)
            sequence_saved = len(tokenized_list) - len(sequence_groups)
            if max(condition_saved, sequence_saved) <= 0:
                resolved_grouping = "no_grouping"
            elif sequence_saved > condition_saved:
                resolved_grouping = "sequence"
            else:
                resolved_grouping = "condition"

        batches: list[_ForwardBatch] = []
        if paired_rows and resolved_grouping == "pair":
            pair_indices = list(range(len(condition_list) // 2))
            pair_limit = 1 if grouping == "serial" else max_pairs_per_forward
            for pair_chunk in self._chunks(pair_indices, pair_limit):
                row_indices = tuple(idx for pair_idx in pair_chunk for idx in (2 * pair_idx, 2 * pair_idx + 1))
                batches.append(
                    _ForwardBatch(
                        indices=row_indices,
                        grouping="pair",
                        group_key_kind="pair_batch",
                        group_key=f"{pair_chunk[0]}:{pair_chunk[-1]}",
                        dataset_flag_shape=(len(pair_chunk), 2),
                        dataset_flag_meaning=(
                            "reference and alternative DNA encoded together; "
                            "one repeated description per pair"
                        ),
                    )
                )
        elif paired_rows and resolved_grouping == "condition":
            for condition_key, pair_indices in pair_condition_groups.items():
                for pair_chunk in self._chunks(pair_indices, max_pairs_per_forward):
                    row_indices = tuple(idx for pair_idx in pair_chunk for idx in (2 * pair_idx, 2 * pair_idx + 1))
                    batches.append(
                        _ForwardBatch(
                            indices=row_indices,
                            grouping="condition",
                            group_key_kind="condition_text",
                            group_key=condition_key,
                            dataset_flag_shape=(1, len(row_indices)),
                            dataset_flag_meaning=(
                                "condition repeated across complete reference/alternative pairs; "
                                "encode description once"
                            ),
                        )
                    )
        elif resolved_grouping == "serial":
            for idx in range(len(tokenized_list)):
                batches.append(
                    _ForwardBatch(
                        indices=(idx,),
                        grouping="serial",
                        group_key_kind="row_index",
                        group_key=str(idx),
                        dataset_flag_shape=(1, 1),
                        dataset_flag_meaning="serial execution; one independent pair per forward",
                    )
                )
        elif resolved_grouping == "no_grouping":
            for chunk in self._chunks(list(range(len(tokenized_list))), max_records_per_forward):
                batches.append(
                    _ForwardBatch(
                        indices=tuple(chunk),
                        grouping="no_grouping",
                        group_key_kind="row_batch",
                        group_key=f"{chunk[0]}:{chunk[-1]}",
                        dataset_flag_shape=(len(chunk), 1),
                        dataset_flag_meaning="batched independent pairs; no sequence or condition shortcut",
                    )
                )
        elif resolved_grouping == "condition":
            for condition_key, indices in condition_groups.items():
                for chunk in self._chunks(indices, max_records_per_forward):
                    batches.append(
                        _ForwardBatch(
                            indices=tuple(chunk),
                            grouping="condition",
                            group_key_kind="condition_text",
                            group_key=condition_key,
                            dataset_flag_shape=(1, len(chunk)),
                            dataset_flag_meaning="condition repeated; encode description once and reuse for DNA rows",
                        )
                    )
        elif resolved_grouping == "sequence":
            for indices in sequence_groups.values():
                first_tokenized = tokenized_list[indices[0]]
                assert first_tokenized is not None
                sequence_name = first_tokenized.source.name or f"sequence_index:{indices[0]}"
                for chunk in self._chunks(indices, max_records_per_forward):
                    batches.append(
                        _ForwardBatch(
                            indices=tuple(chunk),
                            grouping="sequence",
                            group_key_kind="tokenized_sequence",
                            group_key=str(sequence_name),
                            dataset_flag_shape=(1, len(chunk)),
                            dataset_flag_meaning="DNA repeated; encode sequence once and reuse for descriptions",
                        )
                    )
        else:
            raise RuntimeError(f"Unexpected resolved grouping: {resolved_grouping!r}")

        output: list[Prediction | None] = [None] * len(tokenized_list)
        forward_group_id = 0
        extra = dict(extra_provenance or {})
        encoded_by_key = description_cache if description_cache is not None else {}

        def store_predictions(
            batch: _ForwardBatch,
            logits: Any,
            desc_encodings: Sequence[dict[str, Any]],
        ) -> None:
            """Store one forward batch using the resolved retention policy."""

            nonlocal forward_group_id
            for row_pos, (idx, desc_encoded) in enumerate(zip(batch.indices, desc_encodings)):
                tokenized = tokenized_list[idx]
                assert tokenized is not None
                row_logits = logits[row_pos : row_pos + 1]
                expression = self._expression_from_logits(row_logits)
                retained_outputs = (
                    {
                        "expression": expression,
                        **({"logits": row_logits} if retention.logits else {}),
                    }
                    if retention.outputs == "full"
                    else {"expression": _simple_value(expression)}
                )
                prediction = Prediction(
                    sequence=tokenized.source if retention.sequence else None,
                    sequence_name=tokenized.source.name,
                    condition=condition_list[idx],
                    logits=row_logits if retention.logits else None,
                    outputs=retained_outputs,
                    tokens=tokenized if return_tokens and retention.tokens else None,
                    provenance={
                        **self.provenance,
                        **extra,
                        "batch_method": batch_method,
                        "grouping": grouping,
                        "resolved_grouping": batch.grouping,
                        "group_key_kind": batch.group_key_kind,
                        "group_key": batch.group_key,
                        "forward_group_id": forward_group_id,
                        "group_size": len(batch.indices),
                        "dataset_flag_shape": batch.dataset_flag_shape,
                        "dataset_flag_meaning": batch.dataset_flag_meaning,
                        "max_records_per_forward": max_records_per_forward,
                        "max_pairs_per_forward": max_pairs_per_forward,
                        "paired_rows": paired_rows,
                        "prefetch_batches": prefetch_batches,
                        "preprocessing_workers": preprocessing_workers,
                        "preprocessing_backend": preprocessing_backend,
                        "record_index": idx,
                    }
                    if retention.provenance
                    else None,
                    description_tokens=None,
                )
                output[idx] = (
                    self._attach_description_tokens(prediction, desc_encoded)
                    if retention.description_tokens
                    else prediction
                )
            forward_group_id += 1

        def prefetched_batches():
            """Yield batches while the selected CPU backend prepares future work."""

            if not batches:
                return

            sequence_by_key: dict[tuple[Any, ...], TokenizedSequence] = {}
            if preprocessing_workers == 0:
                for batch in batches:
                    tokenized_batch: list[TokenizedSequence] = []
                    desc_encodings = []
                    for idx in batch.indices:
                        if tokenized_list[idx] is None:
                            assert task_list is not None
                            task = task_list[idx]
                            sequence_key = self._sequence_task_key(*task)
                            if sequence_key not in sequence_by_key:
                                sequence_by_key[sequence_key] = self.tokenize_sequence(
                                    task[0],
                                    center=task[1],
                                    strand=task[2],
                                )
                            tokenized_list[idx] = sequence_by_key[sequence_key]
                        tokenized = tokenized_list[idx]
                        assert tokenized is not None
                        tokenized_batch.append(tokenized)

                        desc_key = self._grouping_key_for_condition(condition_list[idx])
                        if desc_key not in encoded_by_key:
                            encoded_by_key[desc_key] = self.tokenize_description(condition_list[idx])
                        desc_encodings.append(encoded_by_key[desc_key])
                    yield batch, tokenized_batch, desc_encodings
                return

            assert preprocessing_executor is not None
            pending: deque[Any] = deque()
            desc_in_flight: dict[str, Future[Any]] = {}
            sequence_in_flight: dict[tuple[Any, ...], Future[Any]] = {}
            next_batch = 0

            def submit(batch: _ForwardBatch) -> None:
                """Submit preprocessing needed by one future model batch."""

                sequence_work = []
                description_work = []
                for idx in batch.indices:
                    sequence_key = None
                    sequence_future = None
                    source = None
                    if tokenized_list[idx] is None:
                        assert task_list is not None
                        task = task_list[idx]
                        sequence_key = self._sequence_task_key(*task)
                        sequence_future = sequence_in_flight.get(sequence_key)
                        if sequence_key not in sequence_by_key and sequence_future is None:
                            if preprocessing_backend == "process":
                                process_payload, source = self._sequence_process_payload(task)
                                sequence_future = executor.submit(
                                    _process_worker_tokenize_sequence,
                                    process_payload,
                                )
                            else:
                                sequence_future = executor.submit(
                                    self.tokenize_sequence,
                                    task[0],
                                    center=task[1],
                                    strand=task[2],
                                )
                            sequence_in_flight[sequence_key] = sequence_future
                    sequence_work.append((sequence_key, sequence_future, source))

                    desc_key = self._grouping_key_for_condition(condition_list[idx])
                    desc_future = None
                    if desc_key not in encoded_by_key:
                        desc_future = desc_in_flight.get(desc_key)
                        if desc_future is None:
                            if preprocessing_backend == "process":
                                description = self.make_description(condition_list[idx])
                                desc_future = executor.submit(_process_worker_tokenize_description, description)
                            else:
                                desc_future = executor.submit(self.tokenize_description, condition_list[idx])
                            desc_in_flight[desc_key] = desc_future
                    description_work.append((desc_key, desc_future))
                pending.append((batch, sequence_work, description_work))

            executor = preprocessing_executor
            initial_count = min(len(batches), prefetch_batches + 1)
            for _ in range(initial_count):
                submit(batches[next_batch])
                next_batch += 1

            while pending:
                batch, sequence_work, description_work = pending.popleft()
                tokenized_batch: list[TokenizedSequence] = []
                for idx, (key, future, source) in zip(batch.indices, sequence_work):
                    if tokenized_list[idx] is None:
                        assert key is not None
                        if key not in sequence_by_key:
                            assert future is not None
                            result = future.result()
                            if preprocessing_backend == "process":
                                assert source is not None
                                result = self._restore_process_tokenized_sequence(result, source)
                            sequence_by_key[key] = result
                            sequence_in_flight.pop(key, None)
                        tokenized_list[idx] = sequence_by_key[key]
                    tokenized = tokenized_list[idx]
                    assert tokenized is not None
                    tokenized_batch.append(tokenized)

                desc_encodings = []
                for key, future in description_work:
                    if key not in encoded_by_key:
                        assert future is not None
                        result = future.result()
                        if preprocessing_backend == "process":
                            result = self._restore_process_description(result)
                        encoded_by_key[key] = result
                        desc_in_flight.pop(key, None)
                    desc_encodings.append(encoded_by_key[key])

                yield batch, tokenized_batch, desc_encodings

                # Refill only after the current GPU forward has finished.
                if next_batch < len(batches):
                    submit(batches[next_batch])
                    next_batch += 1

        stage = progress_description or batch_method.replace("_", " ").title()
        self.logger.info(
            "%s: %d row(s), %d forward batch(es), grouping=%s, preprocessing=%s/%d, prefetch=%d",
            stage,
            len(tokenized_list),
            len(batches),
            resolved_grouping,
            preprocessing_backend,
            preprocessing_workers,
            prefetch_batches,
        )
        try:
            prepared = prefetched_batches()
            for batch, tokenized_batch, desc_encodings in _progress(
                prepared,
                total=len(batches),
                description=stage,
                enabled=show_progress,
                unit="batch",
            ):
                if batch.grouping in {"serial", "no_grouping"}:
                    model_output = self._run_tokenized_no_grouping(tokenized_batch, desc_encodings)
                elif batch.grouping == "pair":
                    model_output = self._run_tokenized_pairs(tokenized_batch, desc_encodings)
                elif batch.grouping == "condition":
                    model_output = self._run_tokenized_repeated_condition(tokenized_batch, desc_encodings[0])
                elif batch.grouping == "sequence":
                    model_output = self._run_tokenized_repeated_sequence(tokenized_batch[0], desc_encodings)
                else:
                    raise RuntimeError(f"Unexpected forward grouping: {batch.grouping!r}")
                store_predictions(batch, self._logits_to_cpu(model_output), desc_encodings)
        finally:
            self.clear_cuda_cache()

        self.logger.info("%s complete", stage)
        return [prediction for prediction in output if prediction is not None]

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
        description_cache: dict[str, dict[str, Any]] | None = None,
    ) -> list[ExpressionPrediction]:
        """Predict expression for sequence/condition rows.

        Accepts many sequences with one condition, one sequence with many
        conditions, or paired sequence/condition rows. Future CPU batches are
        prepared by threads or spawned processes while the GPU runs.
        """

        sequence_list, condition_list = self._sequence_condition_rows(sequences, conditions, condition)
        if not sequence_list:
            return []
        prediction_retention = retention_policy_for(retention).prediction
        with self._preprocessing_executor(
            preprocessing_backend,
            preprocessing_workers,
        ) as preprocessing_executor:
            predictions = self._predict_sequence_tasks(
                [(sequence, center, strand) for sequence in sequence_list],
                condition_list,
                grouping=grouping,
                preprocessing_workers=preprocessing_workers,
                preprocessing_backend=preprocessing_backend,
                preprocessing_executor=preprocessing_executor,
                max_records_per_forward=max_records_per_forward,
                prefetch_batches=prefetch_batches,
                show_progress=show_progress,
                batch_method="predict_multiple_sequences",
                progress_description="Predicting sequences",
                return_tokens=return_tokens,
                retention=prediction_retention,
                description_cache=description_cache,
            )
        return [self._to_expression_prediction(prediction) for prediction in predictions]

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
        """Predict both alleles of one sequence pair using the selected execution mode."""

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
        """Predict reference/alternative pairs with bounded preprocessing.

        ``pair_execution='joint'`` interleaves each reference and alternative
        in one model forward. ``max_pairs_per_forward`` limits variants in that
        mode, so the physical DNA batch has twice as many rows. The default
        ``'separate'`` mode preserves the original two-pass numerical protocol
        and continues to use ``max_records_per_forward``.
        """

        pair_list = list(pairs)
        prediction_retention = retention_policy_for(retention).prediction
        condition_list = self._broadcast_conditions(len(pair_list), conditions, condition)
        if not pair_list:
            return []
        if pair_execution not in {"joint", "separate"}:
            raise ValueError("pair_execution must be 'joint' or 'separate'.")
        if pair_execution == "joint" and max_records_per_forward is not None:
            raise ValueError("Use max_pairs_per_forward when pair_execution='joint'.")
        if pair_execution == "separate" and max_pairs_per_forward is not None:
            raise ValueError("max_pairs_per_forward requires pair_execution='joint'.")
        if pair_execution == "joint" and grouping == "sequence":
            raise ValueError(
                "Joint pair execution does not support grouping='sequence'; "
                "use 'no_grouping', 'condition', 'auto', or pair_execution='separate'."
            )

        centers = [self._pair_center_or_variant(pair, center) for pair in pair_list]
        ref_tasks = [(pair.ref, pair_center, "+") for pair, pair_center in zip(pair_list, centers)]
        alt_tasks = [(pair.alt, pair_center, "+") for pair, pair_center in zip(pair_list, centers)]

        description_cache: dict[str, dict[str, Any]] = {}
        with self._preprocessing_executor(
            preprocessing_backend,
            preprocessing_workers,
        ) as preprocessing_executor:
            if pair_execution == "joint":
                # Adjacent rows form one atomic pair throughout preprocessing,
                # padding, model execution, and output reconstruction.
                joint_tasks = [task for tasks in zip(ref_tasks, alt_tasks) for task in tasks]
                joint_conditions = [
                    item
                    for row_condition in condition_list
                    for item in (row_condition, row_condition)
                ]
                joint_predictions = self._predict_sequence_tasks(
                    joint_tasks,
                    joint_conditions,
                    grouping=grouping,
                    preprocessing_workers=preprocessing_workers,
                    preprocessing_backend=preprocessing_backend,
                    preprocessing_executor=preprocessing_executor,
                    max_records_per_forward=None,
                    prefetch_batches=prefetch_batches,
                    show_progress=show_progress,
                    batch_method="multiple_pairs_joint",
                    progress_description="Predicting allele pairs",
                    return_tokens=True,
                    description_cache=description_cache,
                    paired_rows=True,
                    max_pairs_per_forward=max_pairs_per_forward,
                    retention=prediction_retention,
                )
                ref_predictions = joint_predictions[::2]
                alt_predictions = joint_predictions[1::2]
            else:
                ref_predictions = self._predict_sequence_tasks(
                    ref_tasks,
                    condition_list,
                    grouping=grouping,
                    preprocessing_workers=preprocessing_workers,
                    preprocessing_backend=preprocessing_backend,
                    preprocessing_executor=preprocessing_executor,
                    max_records_per_forward=max_records_per_forward,
                    prefetch_batches=prefetch_batches,
                    show_progress=show_progress,
                    batch_method="multiple_pair_refs",
                    progress_description="Predicting references",
                    return_tokens=True,
                    description_cache=description_cache,
                    retention=prediction_retention,
                )
                alt_predictions = self._predict_sequence_tasks(
                    alt_tasks,
                    condition_list,
                    grouping=grouping,
                    preprocessing_workers=preprocessing_workers,
                    preprocessing_backend=preprocessing_backend,
                    preprocessing_executor=preprocessing_executor,
                    max_records_per_forward=max_records_per_forward,
                    prefetch_batches=prefetch_batches,
                    show_progress=show_progress,
                    batch_method="multiple_pair_alts",
                    progress_description="Predicting alternatives",
                    return_tokens=True,
                    description_cache=description_cache,
                    retention=prediction_retention,
                )
        return [
            PairPrediction(
                ref=self._to_expression_prediction(ref_pred),
                alt=self._to_expression_prediction(alt_pred),
                pair=pair if prediction_retention.sequence else None,
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

    @staticmethod
    def _chunks(indices: Sequence[int], size: int | None):
        """Yield consecutive index chunks, or one unbounded chunk."""

        if size is None:
            yield list(indices)
            return
        if size <= 0:
            raise ValueError("max_records_per_forward must be positive.")
        for start in range(0, len(indices), size):
            yield list(indices[start : start + size])

    @staticmethod
    def _tokenized_sequence_key(tokenized: TokenizedSequence) -> tuple[int, ...]:
        """Return a hashable key for a tokenized DNA row."""

        return tuple(int(value) for value in tokenized.input_ids.tolist())

    @staticmethod
    def _grouping_key_for_condition(condition: Condition) -> str:
        """Return the exact condition text used for grouping and caching."""

        return condition.text()

    def _worker_tokenizer_config(self) -> dict[str, Any]:
        """Build serializable DNA-tokenizer configuration for workers."""

        name_or_path = getattr(self.dna_tokenizer, "name_or_path", None) or self.provenance.get("dna_tokenizer")
        if not name_or_path:
            raise ValueError(
                "Process preprocessing requires a reloadable DNA tokenizer path."
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

    def _worker_description_tokenizer_config(self) -> dict[str, Any]:
        """Build serializable description-tokenizer configuration for workers."""

        name_or_path = getattr(self.desc_tokenizer, "name_or_path", None) or self.provenance.get("description_tokenizer")
        if not name_or_path:
            raise ValueError(
                "Process preprocessing requires a reloadable description tokenizer path."
            )
        return {
            "name_or_path": str(name_or_path),
            "padding_side": getattr(self.desc_tokenizer, "padding_side", "left"),
            "desc_max_seq_len": self.desc_max_seq_len,
        }

