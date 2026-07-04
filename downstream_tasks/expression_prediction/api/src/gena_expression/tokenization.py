"""Centered tokenization matching the attached model inference path."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .sequences import AnnotatedSequence, Feature


@dataclass
class TokenizedSequence:
    """DNA tokens plus token-to-sequence coordinate records."""

    input_ids: Any
    attention_mask: Any
    tokens: list[dict[str, Any]]
    source: AnnotatedSequence
    center: int
    strand: str = "+"

    def as_model_inputs(self) -> dict[str, Any]:
        """Return tensors using the model's expected DNA input names."""

        return {
            "dna_input_ids": self.input_ids,
            "dna_attention_mask": self.attention_mask,
        }

    def token_at_sequence_position(self, position: int) -> int | None:
        """Return the model input position of the token covering ``position``."""

        for row in self.tokens:
            if int(row["start"]) <= position < int(row["end"]):
                return int(row["input_position"])
        return None

    def to_frame(self):
        """Return token records as a pandas DataFrame."""

        import pandas as pd

        return pd.DataFrame(self.tokens)


class CenteredTokenizer:
    """Tokenize a sequence around an anchor using the attached upstream/downstream split."""

    def __init__(
        self,
        dna_tokenizer: Any,
        dna_max_seq_len: int,
        token_len_for_fetch: int,
        num_before: int,
        cls_id: int | None = None,
        sep_id: int | None = None,
        pad_id: int | None = None,
    ) -> None:
        self.dna_tokenizer = dna_tokenizer
        self.dna_max_seq_len = int(dna_max_seq_len)
        self.dna_max_seq_tokens = self.dna_max_seq_len - 2
        self.token_len_for_fetch = int(token_len_for_fetch)
        self.num_before = int(num_before)
        self.cls_id = dna_tokenizer.cls_token_id if cls_id is None else cls_id
        self.sep_id = dna_tokenizer.sep_token_id if sep_id is None else sep_id
        self.pad_id = dna_tokenizer.pad_token_id if pad_id is None else pad_id
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def reverse_complement(sequence: str) -> str:
        """Return the reverse complement of a DNA sequence."""

        complement = str.maketrans("ACGTN", "TGCAN")
        return sequence.translate(complement)[::-1]

    def _records_from_encoded(
        self,
        input_ids: list[int],
        offsets: list[tuple[int, int]],
        genomic_side: str,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx, (start_i, end_i) in enumerate(offsets):
            token_id = int(input_ids[idx])

            # This mirrors the attached CoverageTrackScorer. Some DNA tokenizers
            # use continuation-like tokens where offset start is not enough to
            # recover the token span length.
            if token_id == 5:
                previous_end = int(offsets[idx - 1][1]) if idx > 0 else 0
                length = int(end_i) - previous_end
            else:
                length = int(end_i) - int(start_i)

            records.append(
                {
                    "token_id": token_id,
                    "token": self.dna_tokenizer.decode([token_id]),
                    "length": int(length),
                    "genomic_side": genomic_side,
                }
            )
        return records

    @staticmethod
    def _to_tensor(values: list[int]):
        import torch

        return torch.tensor(values, dtype=torch.long)

    @staticmethod
    def _as_annotated(sequence: AnnotatedSequence | str) -> AnnotatedSequence:
        if isinstance(sequence, AnnotatedSequence):
            return sequence
        return AnnotatedSequence(str(sequence))

    def _resolve_center(self, sequence: AnnotatedSequence, center: int | str | Feature) -> int:
        if isinstance(center, int):
            return center
        if isinstance(center, Feature):
            return (center.start + center.end) // 2
        if center in {"middle", "midpoint"}:
            return len(sequence) // 2
        return sequence.resolve_center(center)

    def tokenize(
        self,
        sequence: AnnotatedSequence | str,
        *,
        center: int | str | Feature = "tss",
        strand: str = "+",
        return_offsets: bool = True,
    ) -> TokenizedSequence:
        """Tokenize a DNA sequence while preserving token coordinate spans.

        The model token order and selection behavior are copied from the
        attached benchmark/scorer code: keep ``num_before`` upstream tokens,
        then fill the remaining token budget downstream, with CLS/SEP added
        outside those DNA tokens.
        """

        source = self._as_annotated(sequence)
        sequence_text = source.sequence.upper()
        reverse = strand == "-"
        center_index = self._resolve_center(source, center)
        records: list[dict[str, Any]] = []

        if self.num_before > 0:
            fetch_len = self.num_before * self.token_len_for_fetch
            if not reverse:
                upstream_seq = sequence_text[max(0, center_index - fetch_len) : center_index]
                genomic_side = "left"
            else:
                upstream_seq = sequence_text[center_index : min(len(sequence_text), center_index + fetch_len)]
                upstream_seq = self.reverse_complement(upstream_seq)
                genomic_side = "right"

            encoded = self.dna_tokenizer.encode_plus(
                upstream_seq,
                return_offsets_mapping=return_offsets,
            )
            upstream_ids = encoded["input_ids"][1:-1]
            upstream_offsets = encoded.get("offset_mapping", [(0, 1)] * len(upstream_ids))[1:-1]

            selected_ids = upstream_ids[-self.num_before :]
            selected_offsets = upstream_offsets[-self.num_before :]
            records.extend(self._records_from_encoded(selected_ids, selected_offsets, genomic_side=genomic_side))

        if not reverse:
            downstream_seq = sequence_text[center_index:]
            genomic_side = "right"
        else:
            downstream_seq = sequence_text[:center_index]
            downstream_seq = self.reverse_complement(downstream_seq)
            genomic_side = "left"

        encoded = self.dna_tokenizer.encode_plus(
            downstream_seq,
            return_offsets_mapping=return_offsets,
        )
        downstream_ids = encoded["input_ids"][1:-1]
        downstream_offsets = encoded.get("offset_mapping", [(0, 1)] * len(downstream_ids))[1:-1]
        downstream_limit = self.dna_max_seq_tokens - self.num_before
        selected_ids = downstream_ids[:downstream_limit]
        selected_offsets = downstream_offsets[:downstream_limit]
        records.extend(self._records_from_encoded(selected_ids, selected_offsets, genomic_side=genomic_side))

        if reverse:
            records.reverse()

        if not records:
            raise ValueError("Tokenization produced no DNA tokens.")

        left_length = sum(record["length"] for record in records if record["genomic_side"] == "left")
        current = center_index - left_length
        for input_position, record in enumerate(records, start=1):
            length = int(record["length"])
            record["input_position"] = input_position
            record["start"] = int(current)
            record["end"] = int(current + length)
            record["center"] = float(current + length / 2)
            record["feature_names"] = [
                feature.name
                for feature in source.features_overlapping(int(current), int(current + length))
            ]
            record["source"] = source.name
            current += length

        input_ids = [self.cls_id] + [record["token_id"] for record in records] + [self.sep_id]
        attention_mask = [1] * len(input_ids)
        return TokenizedSequence(
            input_ids=self._to_tensor(input_ids),
            attention_mask=self._to_tensor(attention_mask),
            tokens=records,
            source=source,
            center=center_index,
            strand=strand,
        )
