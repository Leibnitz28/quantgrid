"""Property tests for canonical JSON."""

from __future__ import annotations

import json
import random
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from quantgrid.domain.hashing import canonical_json, canonical_sha256, is_sha256_hex

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**63), max_value=2**63)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=12),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(max_size=8), children, max_size=5)
    ),
    max_leaves=25,
)


def _shuffle(obj: Any, rng: random.Random) -> Any:
    if isinstance(obj, dict):
        items = [(k, _shuffle(v, rng)) for k, v in obj.items()]
        rng.shuffle(items)
        return dict(items)
    if isinstance(obj, list):
        return [_shuffle(v, rng) for v in obj]
    return obj


def _fold_zero(obj: Any) -> Any:
    if isinstance(obj, float) and obj == 0.0:
        return 0.0
    if isinstance(obj, dict):
        return {k: _fold_zero(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fold_zero(v) for v in obj]
    return obj


@settings(max_examples=200, deadline=None)
@given(json_values)
def test_roundtrip_equal(obj: Any) -> None:
    assert json.loads(canonical_json(obj)) == _fold_zero(obj)


@settings(max_examples=200, deadline=None)
@given(json_values, st.integers(0, 2**31))
def test_stable_under_reordering(obj: Any, seed: int) -> None:
    shuffled = _shuffle(obj, random.Random(seed))
    assert canonical_json(shuffled) == canonical_json(obj)
    assert canonical_sha256(shuffled) == canonical_sha256(obj)


@settings(max_examples=200, deadline=None)
@given(json_values)
def test_idempotent_and_hex(obj: Any) -> None:
    text = canonical_json(obj)
    assert canonical_json(json.loads(text)) == text
    assert is_sha256_hex(canonical_sha256(obj))


@settings(max_examples=100, deadline=None)
@given(json_values)
def test_ascii_only_no_whitespace_padding(obj: Any) -> None:
    text = canonical_json(obj)
    assert text.isascii()
    assert "\n" not in text


@settings(max_examples=100, deadline=None)
@given(json_values, json_values)
def test_distinct_canonical_forms_distinct_hashes(a: Any, b: Any) -> None:
    if canonical_json(a) != canonical_json(b):
        assert canonical_sha256(a) != canonical_sha256(b)
