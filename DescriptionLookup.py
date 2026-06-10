import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


class DescriptionLookup:
    """Deterministically choose one matching JSON description per requested text value."""

    def __init__(
        self,
        json_dir: str | Path,
        keys: Iterable[str],
        seed: int | str = 0,
        **filters: Any,
    ) -> None:
        self.json_dir = Path(json_dir)
        self.keys = list(keys)
        self.seed = str(seed)
        self.filters = filters
        self._lookup: Dict[str, Dict[str, Any]] = {}

        if not self.json_dir.exists():
            raise FileNotFoundError(f"JSON directory does not exist: {self.json_dir}")

        records = list(self._iter_matching_records())
        if not records:
            raise ValueError("No JSON files matched the provided filters")

        for key in self.keys:
            candidates = [
                (path, content)
                for path, content in records
                if self._contains_text(content, key)
            ]
            if not candidates:
                #raise KeyError(f"No matching JSON file contains text {key!r}")
                continue

            index = self._stable_index(key, len(candidates))
            self._lookup[key] = candidates[index][1]

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self._lookup[key]

    def __contains__(self, key: str) -> bool:
        return key in self._lookup

    def keys(self):
        return self._lookup.keys()

    def items(self):
        return self._lookup.items()

    def values(self):
        return self._lookup.values()

    def _iter_matching_records(self):
        for path in sorted(self.json_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                content = json.load(f)

            if self._matches_filters(content):
                yield path, content

    def _matches_filters(self, content: Mapping[str, Any]) -> bool:
        return all(self._get_field(content, field) == value for field, value in self.filters.items())

    @staticmethod
    def _get_field(content: Mapping[str, Any], field: str) -> Optional[Any]:
        current: Any = content
        for part in field.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        return current

    @staticmethod
    def _contains_text(content: Mapping[str, Any], text: str) -> bool:
        json_text = json.dumps(content, ensure_ascii=False).lower()
        return str(text).lower() in json_text

    def _stable_index(self, key: str, n: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).hexdigest()
        return int(digest, 16) % n
