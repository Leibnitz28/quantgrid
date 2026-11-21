"""Unit tests for canonical JSON and hashing."""

from __future__ import annotations

import hashlib
import json

import pytest

from quantgrid.domain.hashing import (
    canonical_json,
    canonical_sha256,
    is_sha256_hex,
    sha256_hex,
)


def test_key_order_invariance() -> None:
    a = {"b": 1, "a": {"y": [1, 2], "x": None}}
    b = {"a": {"x": None, "y": [1, 2]}, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_sha256(a) == canonical_sha256(b)


def test_compact_sorted_output() -> None:
    assert (
        canonical_json({"b": [1, 2.5, None, True], "a": "x"}) == '{"a":"x","b":[1,2.5,null,true]}'
    )


def test_negative_zero_equals_zero() -> None:
    assert canonical_json(-0.0) == canonical_json(0.0) == "0.0"
    assert canonical_sha256({"v": [-0.0]}) == canonical_sha256({"v": [0.0]})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": [bad]})


def test_tuple_equals_list() -> None:
    assert canonical_json({"a": (1, 2)}) == canonical_json({"a": [1, 2]})
    assert canonical_sha256((1, (2, 3))) == canonical_sha256([1, [2, 3]])


@pytest.mark.parametrize("bad_key", [1, None, 1.5, (1,), True])
def test_non_string_keys_rejected(bad_key: object) -> None:
    with pytest.raises(TypeError, match="non-string key"):
        canonical_json({bad_key: 1})


@pytest.mark.parametrize("bad", [b"abc", bytearray(b"abc")])
def test_bytes_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        canonical_json({"x": bad})


@pytest.mark.parametrize("bad", [{1, 2}, object(), 1 + 2j])
def test_other_types_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        canonical_json(bad)


def test_error_path_names_location() -> None:
    with pytest.raises(TypeError, match=r"\$\.a\[1\]"):
        canonical_json({"a": [1, b"x"]})


def test_ascii_escaping_and_unicode_stability() -> None:
    assert canonical_json({"k": "é\u2028"}) == '{"k":"\\u00e9\\u2028"}'
    assert canonical_json("😀") == '"\\ud83d\\ude00"'


def test_bool_is_not_int() -> None:
    assert canonical_json([True, 1]) == "[true,1]"
    assert canonical_sha256(True) != canonical_sha256(1)


def test_int_and_float_differ() -> None:
    # Documented behaviour: spec normalisation (pydantic) must coerce before hashing.
    assert canonical_json(1) == "1"
    assert canonical_json(1.0) == "1.0"


def test_big_int_preserved() -> None:
    assert canonical_json(2**63 - 1) == "9223372036854775807"


def test_golden_digests() -> None:
    assert canonical_json({"a": 1}) == '{"a":1}'
    assert canonical_sha256({"a": 1}) == (
        "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
    )
    assert canonical_sha256([1, 2, 3]) == (
        "a615eeaee21de5179de080de8c3052c8da901138406ba71c38c032845f7d54f4"
    )
    assert sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_hex_str_and_bytes_agree() -> None:
    assert (
        sha256_hex("héllo")
        == sha256_hex("héllo".encode())
        == hashlib.sha256("héllo".encode()).hexdigest()
    )


def test_canonical_sha256_is_sha_of_canonical_json() -> None:
    obj = {"z": [1.5, {"q": None}]}
    assert canonical_sha256(obj) == sha256_hex(canonical_json(obj))


def test_is_sha256_hex() -> None:
    good = "a" * 64
    assert is_sha256_hex(good)
    assert not is_sha256_hex(good.upper())
    assert not is_sha256_hex("0" * 63)
    assert not is_sha256_hex("0" * 65)
    assert not is_sha256_hex("g" * 64)
    assert not is_sha256_hex(None)
    assert not is_sha256_hex(b"0" * 64)
    assert not is_sha256_hex(0)


def test_is_sha256_hex_rejects_trailing_newline() -> None:
    assert not is_sha256_hex("0" * 64 + "\n")


def test_output_is_valid_json() -> None:
    obj = {"a": [1, 2.25, "x", None, {"b": False}]}
    assert json.loads(canonical_json(obj)) == obj
