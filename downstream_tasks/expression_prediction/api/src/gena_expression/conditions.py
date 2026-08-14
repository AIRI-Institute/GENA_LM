"""Biological condition objects and deterministic metadata lookup."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Iterable, Mapping, Optional, Sequence


DescriptionFormatter = Callable[..., str]
ResolvedDescriptionFormatter = Callable[[Mapping[str, Any]], str]
DescriptionFormatterSpec = DescriptionFormatter | str


@contextmanager
def _temporary_sys_path(path: Path):
    """Temporarily make sibling imports available while loading a user file."""

    path_str = str(path)
    added = path_str not in sys.path
    if added:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if added:
            sys.path.remove(path_str)


def _load_static_description_formatter(reference: str) -> DescriptionFormatter:
    """Load ``file.py::ClassName::static_method`` from trusted user code."""

    parts = reference.rsplit("::", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Description formatter references must use "
            "'/path/to/file.py::ClassName::static_method_name'."
        )

    file_name, class_name, method_name = parts
    file_path = Path(file_name).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Description formatter file does not exist: {file_path}")

    module_hash = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_gena_expression_description_formatter_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load description formatter module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        with _temporary_sys_path(file_path.parent):
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise

    try:
        owner = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"{class_name!r} not found in {file_path}") from exc

    try:
        descriptor = inspect.getattr_static(owner, method_name)
    except AttributeError as exc:
        raise ImportError(
            f"{method_name!r} not found on {class_name!r} in {file_path}"
        ) from exc
    if not isinstance(descriptor, staticmethod):
        raise TypeError(
            f"{reference!r} must point to a method declared with @staticmethod."
        )

    formatter = getattr(owner, method_name)
    if not callable(formatter):
        raise TypeError(f"Resolved description formatter is not callable: {reference!r}")
    return formatter


def _description_formatter_source(formatter: DescriptionFormatterSpec) -> str:
    """Return a compact provenance label for a formatter specification."""

    if isinstance(formatter, str):
        return formatter
    stored_source = getattr(formatter, "__gena_expression_formatter_source__", None)
    if stored_source is not None:
        return str(stored_source)
    module = getattr(formatter, "__module__", None)
    qualname = getattr(formatter, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(formatter)


def resolve_description_formatter(
    formatter: DescriptionFormatterSpec,
) -> ResolvedDescriptionFormatter:
    """Resolve and adapt a user formatter to the internal one-mapping contract.

    The metadata mapping is converted to a built-in ``dict`` and passed as the
    first argument. Every additional fixed positional or keyword-only argument
    receives ``None``. Variadic ``*args`` and ``**kwargs`` receive no invented
    values.
    """

    if getattr(formatter, "__gena_expression_resolved_formatter__", False):
        return formatter  # type: ignore[return-value]
    if isinstance(formatter, str):
        source = formatter
        target = _load_static_description_formatter(formatter)
    elif callable(formatter):
        source = _description_formatter_source(formatter)
        target = formatter
    else:
        raise TypeError(
            "description_formatter must be a callable or a static-method reference string."
        )

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise TypeError("Could not inspect the description formatter signature.") from exc

    fixed_positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if not fixed_positional:
        raise TypeError(
            "Description formatter must accept metadata as its first positional argument."
        )
    keyword_only = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    extra_positional_count = len(fixed_positional) - 1

    @wraps(target)
    def render(meta: Mapping[str, Any]) -> str:
        result = target(
            dict(meta),
            *([None] * extra_positional_count),
            **{name: None for name in keyword_only},
        )
        if not isinstance(result, str):
            raise TypeError(
                "Description formatter must return str, "
                f"got {type(result).__name__}."
            )
        return result

    render.__gena_expression_resolved_formatter__ = True  # type: ignore[attr-defined]
    render.__gena_expression_formatter_source__ = source  # type: ignore[attr-defined]
    return render


@dataclass(frozen=True)
class Condition:
    """Description and structured metadata used to condition a model."""

    _description_formatter: ClassVar[ResolvedDescriptionFormatter | None] = None
    _description_formatter_source: ClassVar[str | None] = None

    name: str
    description: str | Mapping[str, Any]
    assay: str | None = None
    cell_type: str | None = None
    tissue: str | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def set_description_formatter(cls, formatter: DescriptionFormatterSpec | None) -> None:
        """Set the description formatter for this Python runtime.

        Passing ``None`` clears the runtime formatter. Models snapshot the
        resolved callable when they are constructed.
        """

        if formatter is None:
            cls.clear_description_formatter()
            return
        resolved = resolve_description_formatter(formatter)
        cls._description_formatter = resolved
        cls._description_formatter_source = _description_formatter_source(resolved)

    @classmethod
    def clear_description_formatter(cls) -> None:
        """Clear the runtime formatter so structured descriptions fail fast."""

        cls._description_formatter = None
        cls._description_formatter_source = None

    @classmethod
    def get_description_formatter(cls) -> ResolvedDescriptionFormatter | None:
        """Return the configured runtime formatter, if any."""

        return cls._description_formatter

    @classmethod
    def get_description_formatter_source(cls) -> str | None:
        """Return the provenance label for the runtime formatter, if any."""

        return cls._description_formatter_source

    def text(self) -> str:
        """Return the exact natural-language text passed to the description tokenizer."""

        if isinstance(self.description, str):
            return self.description
        formatter = self.get_description_formatter()
        if formatter is None:
            raise ValueError(
                "No description formatter is configured. Call "
                "Condition.set_description_formatter(...) first."
            )
        return formatter(self.description)

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
