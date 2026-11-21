"""Canonical JSON serialisation and content hashing.

Every hash in QuantGrid (``spec_hash``, ``spec_sha``, ``dataset_sha``, ``output_sha``,
``code_sha``, ``env_sha``) is a lowercase hex SHA-256 digest.

Canonical JSON means: UTF-8, keys sorted, no insignificant whitespace, ASCII escaping,
finite floats only, ``-0.0`` normalised to ``0.0`` and tuples serialised as lists.
Two semantically identical specs therefore hash identically regardless of the key
order or formatting of the request that produced them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def _normalise(obj: Any, path: str) -> JsonValue:
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float at {path}: {obj!r}")
        return 0.0 if obj == 0.0 else obj  # folds -0.0 into 0.0
    if isinstance(obj, Mapping):
        out: dict[str, JsonValue] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string key at {path}: {key!r}")
            out[key] = _normalise(value, f"{path}.{key}")
        return out
    if isinstance(obj, Sequence) and not isinstance(obj, bytes | bytearray):
        return [_normalise(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    raise TypeError(f"value of type {type(obj).__name__} at {path} is not JSON-canonicalisable")


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to canonical JSON (see module docstring)."""
    return json.dumps(
        _normalise(obj, "$"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_hex(data: bytes | str) -> str:
    """SHA-256 of raw bytes (``str`` is UTF-8 encoded first)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(obj: Any) -> str:
    """SHA-256 of the canonical JSON form of ``obj``."""
    return sha256_hex(canonical_json(obj))


def is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
