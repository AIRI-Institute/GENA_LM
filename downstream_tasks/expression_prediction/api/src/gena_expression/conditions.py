"""Biological condition objects and deterministic metadata lookup."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Condition:
    """Description and structured metadata used to condition a model."""

    name: str
    description: str | Mapping[str, Any]
    assay: str | None = None
    cell_type: str | None = None
    tissue: str | None = None
    metadata: Mapping[str, Any] | None = None

    def text(self) -> str:
        """Return the exact natural-language text passed to the description tokenizer."""

        if isinstance(self.description, str):
            return self.description
        return metadata_to_description(self.description)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the condition to a JSON-friendly dictionary."""

        return {
            "name": self.name,
            "description": self.description,
            "assay": self.assay,
            "cell_type": self.cell_type,
            "tissue": self.tissue,
            "metadata": dict(self.metadata or {}),
        }


def metadata_to_description(meta: Mapping[str, Any]) -> str:
    """Convert structured metadata to the sentence format used by the attached code."""

    line_texts = []
    for key, value in meta.items():
        clean_key = str(key).replace("_", " ")
        clean_value = str(value).replace("_", " ")
        clean_key = re.sub(
            r"^(Characteristics|Chracteristics|Charateristics|Parameter)\s*",
            "",
            clean_key,
        )
        clean_key = re.sub(r"\[|\]", "", clean_key).strip()
        clean_key = clean_key if clean_key else str(key)
        clean_value = clean_value.replace('"', "").strip()
        line_texts.append(f"{clean_key} is {clean_value}.")
    return " ".join(line_texts)


class DescriptionLookup:
    """Deterministically choose one matching JSON description per requested text value.

    By default this preserves the attached helper's behavior: each requested key
    is searched as a case-insensitive substring over the full serialized JSON.
    Dotted-field filters are applied as exact matches before candidate selection.
    Pass ``filters={...}`` or plain keyword fields for global filters. Pass a
    mapping under a requested key, such as ``K562={"id": "ENCFF578UUD"}``, for
    key-specific filters.
    """

    def __init__(
        self,
        json_dir: str | Path,
        keys: Iterable[str],
        seed: int | str = 0,
        strict: bool = False,
        search_fields: Sequence[str] | None = None,
        **filters: Any,
    ) -> None:
        """Index matching description JSON files for deterministic lookup."""

        self.json_dir = Path(json_dir)
        self._keys = list(keys)
        self.seed = str(seed)
        self.filters, self.key_filters = self._split_filters(filters)
        self.strict = bool(strict)
        self.search_fields = tuple(search_fields) if search_fields is not None else None
        self._lookup: Dict[str, Dict[str, Any]] = {}
        self._candidates: dict[str, list[tuple[Path, Dict[str, Any]]]] = {}

        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON directory does not exist: {self.json_dir}")

        records = list(self._iter_matching_records())
        if not records:
            raise ValueError("No JSON files matched the provided filters")

        for key in self._keys:
            candidates = [
                (path, content)
                for path, content in records
                if self._matches_filters(content, self.key_filters.get(key))
                and self._contains_text(content, key, self.search_fields)
            ]
            self._candidates[key] = candidates

            if not candidates:
                if self.strict:
                    raise KeyError(f"No matching JSON file contains text {key!r}")
                continue

            index = self._stable_index(key, len(candidates))
            self._lookup[key] = candidates[index][1]

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Return the selected metadata record for ``key``."""

        return self._lookup[key]

    def __contains__(self, key: str) -> bool:
        """Return whether ``key`` has a selected metadata record."""

        return key in self._lookup

    def keys(self):
        """Return condition keys that had at least one selected match."""

        return self._lookup.keys()

    def items(self):
        """Return selected ``(key, metadata)`` pairs."""

        return self._lookup.items()

    def values(self):
        """Return selected metadata values."""

        return self._lookup.values()

    def candidates(self, key: str) -> list[Path]:
        """Return candidate JSON files considered for ``key``."""

        return [path for path, _ in self._candidates.get(key, [])]

    def condition(self, key: str, *, assay: str | None = None) -> Condition:
        """Return a :class:`Condition` object for one selected metadata record."""

        meta = self[key]
        return Condition(name=key, description=meta, assay=assay, cell_type=key, metadata={"json_dir": str(self.json_dir)})

    def _split_filters(self, filters: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Split constructor keyword filters into global and per-key filters."""

        remaining = dict(filters)
        global_filters: dict[str, Any] = {}
        key_filters: dict[str, dict[str, Any]] = {}

        explicit_global = remaining.pop("filters", None)
        if explicit_global is not None:
            if not isinstance(explicit_global, Mapping):
                raise TypeError("filters=... must be a mapping of metadata fields to values.")
            global_filters.update(dict(explicit_global))

        explicit_key_filters = remaining.pop("key_filters", None)
        if explicit_key_filters is not None:
            if not isinstance(explicit_key_filters, Mapping):
                raise TypeError("key_filters=... must map lookup keys to filter mappings.")
            for key, value in explicit_key_filters.items():
                if not isinstance(value, Mapping):
                    raise TypeError(f"key_filters[{key!r}] must be a mapping.")
                key_filters[str(key)] = dict(value)

        for key in self._keys:
            value = remaining.get(key)
            if isinstance(value, Mapping):
                key_filters[key] = {**key_filters.get(key, {}), **dict(value)}
                remaining.pop(key)

        global_filters.update(remaining)
        return global_filters, key_filters

    def _iter_matching_records(self):
        """Yield JSON records that satisfy the global filters."""

        for path in sorted(self.json_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                content = json.load(handle)

            if self._matches_filters(content):
                yield path, content

    def _matches_filters(self, content: Mapping[str, Any], filters: Mapping[str, Any] | None = None) -> bool:
        """Return whether ``content`` satisfies exact dotted-field filters."""

        active_filters = self.filters if filters is None else filters
        return all(self._get_field(content, field) == value for field, value in active_filters.items())

    @staticmethod
    def _get_field(content: Mapping[str, Any], field: str) -> Optional[Any]:
        """Read a dotted field path from nested metadata."""

        current: Any = content
        for part in field.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    @classmethod
    def _contains_text(
        cls,
        content: Mapping[str, Any],
        text: str,
        search_fields: Sequence[str] | None = None,
    ) -> bool:
        """Return whether selected metadata text contains ``text``."""

        if search_fields is None:
            json_text = json.dumps(content, ensure_ascii=False).lower()
            return str(text).lower() in json_text

        needle = str(text).lower()
        for field in search_fields:
            value = cls._get_field(content, field)
            if value is not None and needle in json.dumps(value, ensure_ascii=False).lower():
                return True
        return False

    def _stable_index(self, key: str, n: int) -> int:
        """Choose a seed-stable candidate index for ``key``."""

        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).hexdigest()
        return int(digest, 16) % n
