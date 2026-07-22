"""Batch planning and preprocessing workers used by model inference."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from ..sequences import AnnotatedSequence, Feature
from .tokenization import CenteredTokenizer


_PROCESS_WORKER_TOKENIZER = None
_PROCESS_WORKER_DESC_TOKENIZER = None
_PROCESS_WORKER_DESC_MAX_SEQ_LEN = None
_THREAD_WORKER_STATE = threading.local()

GroupingMode = Literal["serial", "no_grouping", "condition", "sequence", "auto"]
PairExecutionMode = Literal["joint", "separate"]
PreprocessingBackend = Literal["thread", "process"]


@dataclass(frozen=True)
class _ForwardBatch:
    """One model forward call and the rows it produces."""

    indices: tuple[int, ...]
    grouping: str
    group_key_kind: str
    group_key: str
    dataset_flag_shape: tuple[int, int]
    dataset_flag_meaning: str


def _progress(
    items: Iterable[Any],
    *,
    total: int,
    description: str,
    enabled: bool,
    unit: str,
) -> Iterable[Any]:
    """Wrap meaningful multi-item work in a tqdm progress bar."""

    if not enabled or total < 2:
        return items

    from tqdm.auto import tqdm

    return tqdm(items, total=total, desc=description, unit=unit)


def _process_worker_init(
    tokenizer_config: Mapping[str, Any],
    description_tokenizer_config: Mapping[str, Any],
) -> None:
    """Load process-local preprocessing resources without touching the CUDA model."""

    from transformers import AutoTokenizer

    global _PROCESS_WORKER_TOKENIZER
    global _PROCESS_WORKER_DESC_TOKENIZER
    global _PROCESS_WORKER_DESC_MAX_SEQ_LEN

    dna_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_config["name_or_path"]))
    _PROCESS_WORKER_TOKENIZER = CenteredTokenizer(
        dna_tokenizer=dna_tokenizer,
        dna_max_seq_len=int(tokenizer_config["dna_max_seq_len"]),
        token_len_for_fetch=int(tokenizer_config["token_len_for_fetch"]),
        num_before=int(tokenizer_config["num_before"]),
        cls_id=tokenizer_config.get("cls_id"),
        sep_id=tokenizer_config.get("sep_id"),
        pad_id=tokenizer_config.get("pad_id"),
    )
    _PROCESS_WORKER_DESC_TOKENIZER = AutoTokenizer.from_pretrained(
        str(description_tokenizer_config["name_or_path"]),
        padding_side=str(description_tokenizer_config["padding_side"]),
    )
    _PROCESS_WORKER_DESC_MAX_SEQ_LEN = int(description_tokenizer_config["desc_max_seq_len"])


def _thread_worker_init(
    centered_tokenizer: CenteredTokenizer,
) -> None:
    """Initialize thread-local sequence preprocessing state."""

    _THREAD_WORKER_STATE.centered_tokenizer = centered_tokenizer


def _process_worker_tokenize_sequence(task: Mapping[str, Any]) -> dict[str, Any]:
    """Tokenize one compact DNA task and return a serialization-friendly payload."""

    if _PROCESS_WORKER_TOKENIZER is None:
        raise RuntimeError("Sequence preprocessing worker was not initialized.")
    source = AnnotatedSequence(
        str(task["sequence"]),
        name=task.get("name"),
        features=tuple(Feature(**dict(feature)) for feature in task.get("features", ())),
    )
    tokenized = _PROCESS_WORKER_TOKENIZER.tokenize(
        source,
        center=int(task["center"]),
        strand=str(task["strand"]),
    )
    return {
        "input_ids": tokenized.input_ids.tolist(),
        "attention_mask": tokenized.attention_mask.tolist(),
        "tokens": tokenized.tokens,
        "center": tokenized.center,
        "strand": tokenized.strand,
    }


def _process_worker_tokenize_description(description: str) -> dict[str, Any]:
    """Tokenize one rendered description using process-local tokenizer state."""

    if _PROCESS_WORKER_DESC_TOKENIZER is None or _PROCESS_WORKER_DESC_MAX_SEQ_LEN is None:
        raise RuntimeError("Description preprocessing worker was not initialized.")
    encoding = _PROCESS_WORKER_DESC_TOKENIZER(
        description,
        padding=False,
        truncation=True,
        max_length=_PROCESS_WORKER_DESC_MAX_SEQ_LEN,
    )
    return {
        "description": description,
        "desc_input_ids": list(encoding["input_ids"]),
        "desc_attention_mask": list(encoding["attention_mask"]),
    }


