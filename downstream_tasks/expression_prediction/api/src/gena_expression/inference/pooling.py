"""Expand and reduce model rows for pooled descriptions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..conditions import Condition, PooledCondition


@dataclass(frozen=True)
class _ConditionRow:
    logical_index: int
    condition: Condition
    member_index: int | None = None


def _expand_condition_rows(
    conditions: Sequence[Condition | PooledCondition],
    *,
    paired_rows: bool,
) -> list[_ConditionRow]:
    """Expand one logical row, or one atomic REF/ALT block, at a time."""

    rows = []
    block_size = 2 if paired_rows else 1
    for start in range(0, len(conditions), block_size):
        block = conditions[start : start + block_size]
        condition = block[0]
        if not any(isinstance(item, PooledCondition) for item in block):
            rows.extend(_ConditionRow(start + offset, item) for offset, item in enumerate(block))
            continue
        if not isinstance(condition, PooledCondition) or any(item != condition for item in block):
            raise ValueError("Reference and alternative rows must share the same pooled condition.")
        for member_index, description in enumerate(condition.descriptions):
            member = Condition(
                name=condition.name,
                description=description,
                assay=condition.assay,
                cell_type=condition.cell_type,
                tissue=condition.tissue,
                metadata=condition.metadata,
            )
            rows.extend(_ConditionRow(start + offset, member, member_index) for offset in range(block_size))
    return rows


@dataclass
class _PooledLogits:
    """Accumulate one logical prediction while forward batches are consumed."""

    total: Any = None
    members: set[int] = field(default_factory=set)
    description_tokens: dict[int, dict[str, Any]] = field(default_factory=dict)
    provenance: dict[int, dict[str, Any]] = field(default_factory=dict)

    def add(self, member_index: int, logits: Any, token_count: int) -> None:
        import torch

        if member_index in self.members:
            raise RuntimeError(f"Pool member {member_index} produced more than one output.")
        if logits.ndim != 3 or logits.shape[1] < token_count:
            raise RuntimeError("Pooled logits must cover all positions in the tokenized input.")
        values = logits[:, :token_count, :]
        if self.total is None:
            self.total = values.to(dtype=torch.float32, copy=True)
        else:
            if values.shape != self.total.shape:
                raise RuntimeError("Pooled descriptions produced incompatible logit shapes.")
            self.total.add_(values)
        self.members.add(member_index)
