"""Tests for the deterministic result codec (``quantgrid.results.codec``)."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quantgrid
from quantgrid.domain.hashing import canonical_json
from quantgrid.domain.types import TaskOutput
from quantgrid.results.codec import (
    METADATA_KEYS,
    TS_TYPE,
    DecodedResult,
    decode_result,
    encode_result,
    normalise_metrics,
    validate_artifact,
)

SPEC_HASH = hashlib.sha256(b"task spec").hexdigest()
OTHER_HASH = hashlib.sha256(b"another task spec").hexdigest()
ENGINE = "pairs_zscore@1+qg0.1.0"
REPO_ROOT = Path(__file__).resolve().parents[2]


def make_output(n: int = 500) -> TaskOutput:
    """A realistic, fully deterministic task output (also built in a fresh subprocess)."""
    rng = np.random.default_rng(20261119)
    ts = np.int64(1_700_000_000_000_000_000) + np.arange(n, dtype=np.int64) * np.int64(
        60_000_000_000
    )
    returns = rng.normal(0.0, 0.01, n)
    zscore = np.cumsum(rng.normal(0.0, 1.0, n)) / 10.0
    zscore[:20] = np.nan  # rolling-window warm-up
    position = np.where(zscore > 1.0, -1, np.where(zscore < -1.0, 1, 0)).astype(np.int64)
    equity = np.cumprod(1.0 + returns)
    return TaskOutput(
        metrics={
            "sharpe": float(np.mean(returns) / np.std(returns) * np.sqrt(252.0)),
            "total_return": float(equity[-1] - 1.0),
            "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1.0)),
            "n_trades": int(np.count_nonzero(np.diff(position))),
            "bootstrap_sharpe_p05": None,
        },
        series={
            "ts": ts,
            "zscore": zscore,
            "position": position,
            "returns": returns,
            "equity": equity,
            "in_market": position != 0,
        },
        diagnostics={"warmup_bars": 20, "note": "synthetic", "fill_ratio": 0.5},
    )


def _with(output: TaskOutput, **changes: Any) -> TaskOutput:
    return TaskOutput(
        metrics=changes.get("metrics", output.metrics),
        series=changes.get("series", output.series),
        diagnostics=changes.get("diagnostics", output.diagnostics),
    )


def _rewrite(
    data: bytes, metadata: dict[str, str] | None = None, *, store_schema: bool = False
) -> bytes:
    """Re-write an artifact's table with different footer metadata (tampering helper)."""
    table = pq.read_table(pa.BufferReader(data)).replace_schema_metadata(None)
    sink = pa.BufferOutputStream()
    with pq.ParquetWriter(sink, table.schema, store_schema=store_schema) as writer:
        writer.write_table(table)
        if metadata:
            writer.add_key_value_metadata(metadata)
    return bytes(sink.getvalue().to_pybytes())


def _metadata(data: bytes) -> dict[str, str]:
    """Footer key-value metadata in on-disk order (``read_schema`` preserves the file's
    order; ``ParquetFile.metadata.metadata`` returns a hash-ordered dict)."""
    raw = pq.read_schema(pa.BufferReader(data)).metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def test_encode_is_byte_identical_across_calls() -> None:
    first = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE)
    second = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE)
    assert first.data == second.data
    assert first.sha256 == second.sha256 == hashlib.sha256(first.data).hexdigest()


def test_encode_is_byte_identical_in_a_fresh_interpreter(tmp_path: Path) -> None:
    out_file = tmp_path / "subprocess.parquet"
    code = (
        "import sys\n"
        "from quantgrid.results.codec import encode_result\n"
        "from tests.unit.test_result_codec import ENGINE, SPEC_HASH, make_output\n"
        "data = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE).data\n"
        "open(sys.argv[1], 'wb').write(data)\n"
    )
    src_dir = str(Path(quantgrid.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONHASHSEED": "12345",  # different str-hash seed than this process
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), src_dir]),
    }
    subprocess.run(
        [sys.executable, "-c", code, str(out_file)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=120,
        capture_output=True,
    )
    in_process = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE).data
    assert out_file.read_bytes() == in_process


def test_encode_ignores_mapping_order_and_array_memory_layout() -> None:
    base = make_output()
    reordered_series = {k: base.series[k] for k in reversed(list(base.series))}
    strided = {}
    for name, values in reordered_series.items():
        doubled = np.repeat(values, 2)  # same values, viewed through a non-contiguous slice
        strided[name] = doubled[::2]
        assert not strided[name].flags["C_CONTIGUOUS"] or values.size <= 1
    variant = TaskOutput(
        metrics={k: base.metrics[k] for k in reversed(list(base.metrics))},
        series=strided,
        diagnostics={k: base.diagnostics[k] for k in reversed(list(base.diagnostics))},
    )
    assert (
        encode_result(SPEC_HASH, variant, engine_version=ENGINE).data
        == encode_result(SPEC_HASH, base, engine_version=ENGINE).data
    )


def _float_from_bits(bits: int) -> float:
    value: float = struct.unpack("<d", struct.pack("<Q", bits))[0]
    return value


def test_semantically_equal_floats_encode_identically() -> None:
    base = make_output(50)
    zscore = np.array(base.series["zscore"], copy=True)
    zscore[0] = _float_from_bits(0xFFF8_0000_0000_0001)  # NaN, other sign and payload
    zscore.view(np.uint64)[1] = 0x7FF0_0000_0000_0001  # signalling NaN
    zscore[30] = -0.0
    canonical = np.array(base.series["zscore"], copy=True)
    canonical[30] = 0.0
    assert zscore.view(np.uint64)[0] != canonical.view(np.uint64)[0]
    assert zscore.view(np.uint64)[1] != canonical.view(np.uint64)[1]

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # canonicalisation must not emit FP warnings
        a = encode_result(
            SPEC_HASH, _with(base, series={**base.series, "zscore": zscore}), engine_version=ENGINE
        )
    b = encode_result(
        SPEC_HASH, _with(base, series={**base.series, "zscore": canonical}), engine_version=ENGINE
    )
    assert a.data == b.data
    decoded = decode_result(a.data).series()["zscore"]
    assert set(decoded.view(np.uint64)[np.isnan(decoded)].tolist()) == {0x7FF8_0000_0000_0000}
    assert not np.signbit(decoded[30])


def test_numpy_scalars_and_python_numbers_encode_identically() -> None:
    base = make_output(50)
    numpy_metrics = {
        k: (np.float64(v) if isinstance(v, float) else np.int64(v) if isinstance(v, int) else v)
        for k, v in base.metrics.items()
    }
    numpy_diag = {
        k: (np.int64(v) if isinstance(v, int) else v) for k, v in base.diagnostics.items()
    }
    encoded = encode_result(
        SPEC_HASH, _with(base, metrics=numpy_metrics, diagnostics=numpy_diag), engine_version=ENGINE
    )
    assert encoded.data == encode_result(SPEC_HASH, base, engine_version=ENGINE).data
    assert all(type(v) in (int, float, type(None)) for v in encoded.metrics.values())


def test_datetime64_and_int64_timestamps_encode_identically() -> None:
    base = make_output(50)
    as_datetime = base.series["ts"].astype("datetime64[ns]")
    a = encode_result(
        SPEC_HASH, _with(base, series={**base.series, "ts": as_datetime}), engine_version=ENGINE
    )
    assert a.data == encode_result(SPEC_HASH, base, engine_version=ENGINE).data


def test_artifact_bytes_depend_on_every_input() -> None:
    base = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE).data
    assert encode_result(OTHER_HASH, make_output(), engine_version=ENGINE).data != base
    assert encode_result(SPEC_HASH, make_output(), engine_version=ENGINE + "x").data != base
    out = make_output()
    tweaked = _with(out, metrics={**out.metrics, "sharpe": 0.25})
    assert encode_result(SPEC_HASH, tweaked, engine_version=ENGINE).data != base


# --------------------------------------------------------------------------------------
# Round trip and metadata
# --------------------------------------------------------------------------------------


def test_round_trip() -> None:
    output = make_output()
    encoded = encode_result(SPEC_HASH, output, engine_version=ENGINE)
    decoded = decode_result(encoded.data)

    assert isinstance(decoded, DecodedResult)
    assert decoded.spec_hash == SPEC_HASH
    assert decoded.engine_version == ENGINE
    assert decoded.result_schema_version == 1
    assert decoded.metrics == encoded.metrics == normalise_metrics(output.metrics)
    assert list(decoded.metrics) == sorted(output.metrics)
    assert decoded.diagnostics == dict(output.diagnostics)
    assert decoded.row_count == encoded.row_count == 500

    table = decoded.table
    assert table.column_names == ["ts", "equity", "in_market", "position", "returns", "zscore"]
    assert table.schema.field("ts").type == TS_TYPE
    assert table.schema.field("ts").type == pa.timestamp("ns", tz="UTC")
    assert table.schema.field("equity").type == pa.float64()
    assert table.schema.field("position").type == pa.int64()
    assert table.schema.field("in_market").type == pa.bool_()
    assert all(not field.nullable for field in table.schema)

    series = decoded.series()
    assert list(series) == table.column_names
    assert series["ts"].dtype == np.int64
    for name, values in output.series.items():
        np.testing.assert_array_equal(series[name], values)  # NaN positions must match
    assert np.isnan(series["zscore"][:20]).all()


def test_metadata_contents() -> None:
    output = make_output()
    encoded = encode_result(SPEC_HASH, output, engine_version=ENGINE)
    metadata = _metadata(encoded.data)

    assert list(metadata) == list(METADATA_KEYS)  # exactly these keys, in sorted order
    assert metadata == {
        "quantgrid.diagnostics": canonical_json(output.diagnostics),
        "quantgrid.engine_version": ENGINE,
        "quantgrid.metrics": canonical_json(output.metrics),
        "quantgrid.result_schema_version": "1",
        "quantgrid.spec_hash": SPEC_HASH,
    }
    assert json.loads(metadata["quantgrid.metrics"])["bootstrap_sharpe_p05"] is None
    assert b"ARROW:schema" not in encoded.data  # no pyarrow-specific schema blob


def test_writer_options_are_fixed() -> None:
    encoded = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE)
    meta = pq.ParquetFile(pa.BufferReader(encoded.data)).metadata
    assert meta.num_row_groups == 1
    assert meta.format_version == "2.6"
    for i in range(meta.num_columns):
        column = meta.row_group(0).column(i)
        assert column.compression == "ZSTD"
        assert column.is_stats_set
        assert column.dictionary_page_offset is None  # no dictionary encoding


def test_empty_series_round_trip() -> None:
    output = TaskOutput(
        metrics={"sharpe": None},
        series={"ts": np.array([], dtype=np.int64), "pnl": np.array([], dtype=np.float64)},
    )
    encoded = encode_result(SPEC_HASH, output, engine_version=ENGINE)
    decoded = decode_result(encoded.data)
    assert encoded.row_count == decoded.row_count == 0
    assert decoded.table.column_names == ["ts", "pnl"]
    assert validate_artifact(encoded.data, expected_spec_hash=SPEC_HASH) == []


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), np.float64("nan")])
def test_non_finite_metrics_are_rejected(bad: float) -> None:
    output = _with(make_output(20), metrics={"sharpe": bad})
    with pytest.raises(ValueError, match="finite"):
        encode_result(SPEC_HASH, output, engine_version=ENGINE)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (True, "boolean"),
        (np.bool_(False), "boolean"),
        ("1.0", "unsupported"),
        ([1.0], "unsupported"),
        (2**63, "int64"),
    ],
)
def test_non_numeric_metrics_are_rejected(value: Any, match: str) -> None:
    output = _with(make_output(20), metrics={"sharpe": value})
    with pytest.raises(ValueError, match=match):
        encode_result(SPEC_HASH, output, engine_version=ENGINE)


def test_nan_and_inf_are_allowed_inside_float_series() -> None:
    base = make_output(30)
    returns = np.array(base.series["returns"], copy=True)
    returns[[0, 5, 29]] = np.nan
    returns[7] = np.inf
    encoded = encode_result(
        SPEC_HASH, _with(base, series={**base.series, "returns": returns}), engine_version=ENGINE
    )
    decoded = decode_result(encoded.data).series()["returns"]
    np.testing.assert_array_equal(np.isnan(decoded), np.isnan(returns))
    assert decoded[7] == np.inf
    assert decode_result(encoded.data).table.column("returns").null_count == 0  # NaN != null


@pytest.mark.parametrize(
    ("series", "match"),
    [
        ({"pnl": np.zeros(3)}, "must contain a 'ts'"),
        ({"ts": np.arange(3), "pnl": np.zeros(4)}, "equal length"),
        ({"ts": np.arange(4), "pnl": np.zeros((2, 2))}, "1-D"),
        ({"ts": np.arange(3), "pnl": [0.0, 1.0, 2.0]}, "numpy array"),
        ({"ts": np.arange(3), "pnl": np.zeros(3, dtype=np.complex128)}, "unsupported dtype"),
        ({"ts": np.arange(3), "pnl": np.array(["a", "b", "c"])}, "unsupported dtype"),
        ({"ts": np.arange(3), "pnl": np.array([1, 2, 3], dtype=object)}, "unsupported dtype"),
        ({"ts": np.arange(3, dtype=np.float64)}, "integer dtype"),
        ({"ts": np.arange(3).astype("datetime64[s]")}, "datetime64\\[ns\\]"),
        ({"ts": np.array(["NaT", "2024-01-01"], dtype="datetime64[ns]")}, "NaT"),
        ({"ts": np.array([2**63], dtype=np.uint64)}, "int64"),
        ({"ts": np.arange(3), "bad name": np.zeros(3)}, "invalid series column name"),
        ({"ts": np.arange(3), "": np.zeros(3)}, "invalid series column name"),
        ({"ts": np.arange(2), "x": np.ma.masked_array([1.0, 2.0], mask=[True, False])}, "masked"),
    ],
)
def test_invalid_series_are_rejected(series: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        encode_result(SPEC_HASH, TaskOutput(metrics={}, series=series), engine_version=ENGINE)


def test_narrow_numeric_dtypes_are_widened_exactly() -> None:
    narrow = TaskOutput(
        metrics={},
        series={
            "ts": np.array([1, 2, 3], dtype=np.int32),
            "f": np.array([0.5, 1.5, -2.25], dtype=np.float32),
            "u": np.array([1, 2, 3], dtype=np.uint16),
        },
    )
    wide = TaskOutput(
        metrics={},
        series={
            "ts": np.array([1, 2, 3], dtype=np.int64),
            "f": np.array([0.5, 1.5, -2.25], dtype=np.float64),
            "u": np.array([1, 2, 3], dtype=np.int64),
        },
    )
    assert encode_result(SPEC_HASH, narrow, engine_version=ENGINE).data == (
        encode_result(SPEC_HASH, wide, engine_version=ENGINE).data
    )


def test_byte_order_does_not_change_the_encoding() -> None:
    base = make_output(40)
    swapped = {
        name: values.astype(values.dtype.newbyteorder(">")) if values.dtype.itemsize > 1 else values
        for name, values in base.series.items()
    }
    swapped_dt = {**swapped, "ts": base.series["ts"].astype("datetime64[ns]").astype(">M8[ns]")}
    expected = encode_result(SPEC_HASH, base, engine_version=ENGINE).data
    for series in (swapped, swapped_dt):
        assert encode_result(SPEC_HASH, _with(base, series=series), engine_version=ENGINE).data == (
            expected
        )


@pytest.mark.parametrize("spec_hash", ["", "abc", SPEC_HASH.upper(), SPEC_HASH + "0"])
def test_malformed_spec_hash_is_rejected(spec_hash: str) -> None:
    with pytest.raises(ValueError, match="spec_hash"):
        encode_result(spec_hash, make_output(10), engine_version=ENGINE)


@pytest.mark.parametrize("engine_version", ["", "has space", "x" * 129, "tab\tchar", "é"])
def test_malformed_engine_version_is_rejected(engine_version: str) -> None:
    with pytest.raises(ValueError, match="engine_version"):
        encode_result(SPEC_HASH, make_output(10), engine_version=engine_version)


@pytest.mark.parametrize(
    "metrics", [{"bad-name": 1.0}, {"": 1.0}, {"1starts_with_digit": 1.0}, {3: 1.0}]
)
def test_invalid_metric_names_are_rejected(metrics: dict[Any, Any]) -> None:
    with pytest.raises(ValueError, match="invalid metric name"):
        encode_result(SPEC_HASH, _with(make_output(10), metrics=metrics), engine_version=ENGINE)


@pytest.mark.parametrize(
    "diagnostics", [{"x": float("nan")}, {"x": [1, 2]}, {"x": {"nested": 1}}, {"": 1}]
)
def test_invalid_diagnostics_are_rejected(diagnostics: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        encode_result(
            SPEC_HASH, _with(make_output(10), diagnostics=diagnostics), engine_version=ENGINE
        )


def test_encode_requires_task_output() -> None:
    with pytest.raises(TypeError):
        encode_result(SPEC_HASH, {"metrics": {}}, engine_version=ENGINE)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# decode_result / validate_artifact on bad input
# --------------------------------------------------------------------------------------


def test_validate_accepts_a_good_artifact() -> None:
    encoded = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE)
    assert validate_artifact(encoded.data, expected_spec_hash=SPEC_HASH) == []
    assert (
        validate_artifact(
            encoded.data, expected_spec_hash=SPEC_HASH, expected_metrics=encoded.metrics
        )
        == []
    )


def test_validate_reports_spec_hash_and_metrics_mismatch() -> None:
    encoded = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE)
    problems = validate_artifact(encoded.data, expected_spec_hash=OTHER_HASH)
    assert len(problems) == 1
    assert "spec_hash mismatch" in problems[0]

    other_metrics = {**encoded.metrics, "sharpe": 99.0}
    problems = validate_artifact(
        encoded.data, expected_spec_hash=SPEC_HASH, expected_metrics=other_metrics
    )
    assert len(problems) == 1
    assert "metrics mismatch" in problems[0]

    missing_key = {k: v for k, v in encoded.metrics.items() if k != "bootstrap_sharpe_p05"}
    problems = validate_artifact(
        encoded.data, expected_spec_hash=SPEC_HASH, expected_metrics=missing_key
    )
    assert len(problems) == 1
    assert "metrics mismatch" in problems[0]


def test_metrics_comparison_survives_jsonb_number_rendering() -> None:
    # PostgreSQL JSONB re-renders numbers: the float 1e16 comes back as the integer
    # 10000000000000000 and 2.0 may come back as 2. Equal values must not be flagged.
    output = _with(make_output(20), metrics={"big": 1e16, "two": 2.0, "none": None})
    data = encode_result(SPEC_HASH, output, engine_version=ENGINE).data
    assert '"big":1e+16' in _metadata(data)["quantgrid.metrics"]
    from_db = {"big": 10**16, "two": 2, "none": None}
    assert validate_artifact(data, expected_spec_hash=SPEC_HASH, expected_metrics=from_db) == []
    off_by_one = {"big": 10**16 + 1, "two": 2, "none": None}
    assert validate_artifact(data, expected_spec_hash=SPEC_HASH, expected_metrics=off_by_one) != []
    none_vs_zero = {"big": 10**16, "two": 2, "none": 0}
    assert (
        validate_artifact(data, expected_spec_hash=SPEC_HASH, expected_metrics=none_vs_zero) != []
    )


@pytest.mark.parametrize("data", [b"", b"not parquet at all", b"PAR1" + b"\0" * 64 + b"PAR1"])
def test_garbage_bytes(data: bytes) -> None:
    problems = validate_artifact(data, expected_spec_hash=SPEC_HASH)
    assert len(problems) == 1
    assert "not a readable Parquet file" in problems[0]
    with pytest.raises(ValueError, match="not a readable Parquet file"):
        decode_result(data)


def test_truncated_artifact_is_reported() -> None:
    data = encode_result(SPEC_HASH, make_output(), engine_version=ENGINE).data
    assert validate_artifact(data[: len(data) // 2], expected_spec_hash=SPEC_HASH) != []


def test_parquet_without_quantgrid_metadata() -> None:
    data = encode_result(SPEC_HASH, make_output(20), engine_version=ENGINE).data
    plain = _rewrite(data, None, store_schema=True)
    problems = validate_artifact(plain, expected_spec_hash=SPEC_HASH)
    assert {f"missing metadata key {key!r}" for key in METADATA_KEYS} <= set(problems)
    assert any("unexpected metadata keys ['ARROW:schema']" in p for p in problems)
    with pytest.raises(ValueError, match="missing metadata"):
        decode_result(plain)


def test_tampered_metadata_is_reported() -> None:
    data = encode_result(SPEC_HASH, make_output(20), engine_version=ENGINE).data
    good = _metadata(data)

    nan_metrics = _rewrite(data, {**good, "quantgrid.metrics": '{"sharpe":NaN}'})
    assert any(
        "invalid metrics metadata" in p
        for p in validate_artifact(nan_metrics, expected_spec_hash=SPEC_HASH)
    )
    with pytest.raises(ValueError, match="non-finite"):
        decode_result(nan_metrics)

    pretty = _rewrite(
        data,
        {**good, "quantgrid.metrics": json.dumps(json.loads(good["quantgrid.metrics"]), indent=1)},
    )
    assert validate_artifact(pretty, expected_spec_hash=SPEC_HASH) == [
        "metrics metadata is not canonical JSON"
    ]

    v2 = _rewrite(data, {**good, "quantgrid.result_schema_version": "2"})
    assert validate_artifact(v2, expected_spec_hash=SPEC_HASH) == [
        "result_schema_version is '2', expected '1'"
    ]
    assert decode_result(v2).result_schema_version == 2  # decode is lenient on version

    bad_version = _rewrite(data, {**good, "quantgrid.result_schema_version": "one"})
    with pytest.raises(ValueError, match="not an integer"):
        decode_result(bad_version)

    nested = _rewrite(data, {**good, "quantgrid.diagnostics": '{"x":[1]}'})
    assert any(
        "invalid diagnostics metadata" in p
        for p in validate_artifact(nested, expected_spec_hash=SPEC_HASH)
    )

    extra = _rewrite(data, {**good, "quantgrid.task_id": "1234"})
    assert validate_artifact(extra, expected_spec_hash=SPEC_HASH) == [
        "unexpected metadata keys ['quantgrid.task_id']"
    ]


def test_schema_problems_are_reported() -> None:
    data = encode_result(SPEC_HASH, make_output(20), engine_version=ENGINE).data
    metadata = _metadata(data)
    table = pq.read_table(pa.BufferReader(data)).replace_schema_metadata(None)

    def write(t: pa.Table) -> bytes:
        sink = pa.BufferOutputStream()
        with pq.ParquetWriter(sink, t.schema, store_schema=False) as writer:
            writer.write_table(t)
            writer.add_key_value_metadata(metadata)
        return bytes(sink.getvalue().to_pybytes())

    reordered = table.select(["equity", "ts", *table.column_names[2:]])
    assert any(
        "first column must be 'ts'" in p
        for p in validate_artifact(write(reordered), expected_spec_hash=SPEC_HASH)
    )

    unsorted = table.select(["ts", "zscore", "equity"])
    assert any(
        "unique and sorted" in p
        for p in validate_artifact(write(unsorted), expected_spec_hash=SPEC_HASH)
    )

    with_nulls = table.set_column(
        table.column_names.index("equity"),
        "equity",
        pa.array([None] * table.num_rows, pa.float64()),
    )
    assert any(
        "null" in p for p in validate_artifact(write(with_nulls), expected_spec_hash=SPEC_HASH)
    )

    as_text = table.append_column("label", pa.array(["x"] * table.num_rows))
    assert any(
        "unsupported type" in p
        for p in validate_artifact(write(as_text), expected_spec_hash=SPEC_HASH)
    )

    ms_ts = table.set_column(0, "ts", table.column("ts").cast(pa.timestamp("ms", tz="UTC")))
    assert any(
        "column 'ts' has type" in p
        for p in validate_artifact(write(ms_ts), expected_spec_hash=SPEC_HASH)
    )


def test_validate_rejects_malformed_expected_hash() -> None:
    data = encode_result(SPEC_HASH, make_output(10), engine_version=ENGINE).data
    with pytest.raises(ValueError, match="expected_spec_hash"):
        validate_artifact(data, expected_spec_hash="nope")
