"""Deterministic Parquet encoding of one task result (result schema v1).

An artifact is one Parquet file:

* Columns: ``ts`` first, as ``timestamp[ns, tz=UTC]``, then every other
  ``TaskOutput.series`` key in sorted order. Float columns are float64, integer
  columns int64 and boolean columns bool. Every column is REQUIRED (no nulls). NaN is
  allowed in float columns (e.g. indicator warm-up). Float values are canonicalised:
  every NaN becomes the single quiet-NaN bit pattern and ``-0.0`` becomes ``0.0``, so
  semantically identical results encode identically.
* Footer key-value metadata: exactly these keys, written in this order:

  - ``quantgrid.diagnostics``: canonical JSON object of scalar diagnostics
  - ``quantgrid.engine_version``: caller-supplied engine identifier
  - ``quantgrid.metrics``: canonical JSON object; each value is a finite number or null
  - ``quantgrid.result_schema_version``: ``"1"``
  - ``quantgrid.spec_hash``: SHA-256 of the task spec

Determinism contract: the bytes are a pure function of ``(spec_hash, output,
engine_version)`` and the Parquet library version. The library version appears in the
footer's ``created_by`` and is covered by ``env_sha``. Nothing tied to a time, host,
process or attempt is embedded: no task_id, batch_id, wall-clock time, hostname or
Arrow schema blob. Writer options are fixed: format version, compression and level,
no dictionary encoding, row-group and page sizes, page version, statistics on, no page
index or checksums. So the same task spec yields byte-identical artifacts, and therefore
the same content address, on every attempt and every worker running the same code and
environment. ``engine_version`` must follow the same rule: derive it from the code
version, never from a time or host.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from quantgrid.domain.hashing import canonical_json, is_sha256_hex, sha256_hex
from quantgrid.domain.specs import RESULT_SCHEMA_VERSION
from quantgrid.domain.types import MetricValue, TaskOutput

META_DIAGNOSTICS: Final = "quantgrid.diagnostics"
META_ENGINE_VERSION: Final = "quantgrid.engine_version"
META_METRICS: Final = "quantgrid.metrics"
META_SCHEMA_VERSION: Final = "quantgrid.result_schema_version"
META_SPEC_HASH: Final = "quantgrid.spec_hash"
#: Every metadata key of a v1 artifact, in the order it is written (sorted).
METADATA_KEYS: Final = (
    META_DIAGNOSTICS,
    META_ENGINE_VERSION,
    META_METRICS,
    META_SCHEMA_VERSION,
    META_SPEC_HASH,
)
RESULT_SCHEMA_VERSION_STR: Final = str(RESULT_SCHEMA_VERSION)

TS_COLUMN: Final = "ts"
TS_TYPE: Final = pa.timestamp("ns", tz="UTC")
#: Rows per Parquet row group (fixed, so the layout never depends on the caller).
ROW_GROUP_SIZE: Final = 65_536
#: Fixed Parquet writer options. Changing any of them changes artifact bytes (and the
#: code_sha), so byte-for-byte comparisons only hold within one code version.
WRITER_OPTIONS: Final[Mapping[str, Any]] = {
    "version": "2.6",
    "compression": "zstd",
    "compression_level": 3,
    "use_dictionary": False,
    "write_statistics": True,
    "data_page_size": 1 << 20,
    "data_page_version": "1.0",
    "use_byte_stream_split": False,
    "store_schema": False,
    "write_page_index": False,
    "write_page_checksum": False,
}

#: Series column and metric names: identifier-like, so they are safe as SQL/CSV columns.
NAME_PATTERN: Final = r"[A-Za-z_][A-Za-z0-9_]{0,63}"
_NAME_RE = re.compile(NAME_PATTERN)
_ENGINE_VERSION_RE = re.compile(r"[\x21-\x7e]{1,128}")  # visible ASCII, no whitespace
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1

DiagnosticValue = float | int | str | bool | None


@dataclass(frozen=True, slots=True)
class EncodedResult:
    """Encoded artifact bytes plus the normalised metrics to store in
    ``result_artifacts.metrics``. Metric keys are sorted, values are plain ``int``,
    finite ``float`` or ``None``."""

    data: bytes
    row_count: int
    metrics: dict[str, MetricValue]

    @property
    def sha256(self) -> str:
        """Content address of :attr:`data` (what the artifact store will compute)."""
        return sha256_hex(self.data)


@dataclass(frozen=True, slots=True)
class DecodedResult:
    """A decoded artifact. ``table`` keeps the file's column order and types."""

    spec_hash: str
    metrics: dict[str, MetricValue]
    diagnostics: dict[str, DiagnosticValue]
    engine_version: str
    result_schema_version: int
    table: pa.Table

    @property
    def row_count(self) -> int:
        return int(self.table.num_rows)

    def series(self) -> dict[str, npt.NDArray[np.generic]]:
        """Columns as numpy arrays (``ts`` as int64 nanoseconds since the epoch, UTC)."""
        out: dict[str, npt.NDArray[np.generic]] = {}
        for name in self.table.column_names:
            column = self.table.column(name)
            if name == TS_COLUMN:
                column = column.cast(pa.int64())
            out[str(name)] = np.asarray(column.to_numpy())
        return out


# --------------------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------------------


def encode_result(spec_hash: str, output: TaskOutput, *, engine_version: str) -> EncodedResult:
    """Encode ``output`` as a deterministic Parquet artifact (see module docstring).

    Raises ``ValueError`` on invalid input: a malformed ``spec_hash`` or
    ``engine_version``; series that are missing ``ts``, not 1-D numpy arrays, of unequal
    length or of an unsupported dtype; bad column or metric names; metric values that
    are not a finite number or ``None``; diagnostics that are not finite scalars.
    """
    if not isinstance(output, TaskOutput):
        raise TypeError(f"expected TaskOutput, got {type(output).__name__}")
    if not is_sha256_hex(spec_hash):
        raise ValueError(f"spec_hash must be a lowercase hex SHA-256 digest, got {spec_hash!r}")
    if not isinstance(engine_version, str) or not _ENGINE_VERSION_RE.fullmatch(engine_version):
        raise ValueError(
            "engine_version must be 1-128 visible ASCII characters without whitespace, "
            f"got {engine_version!r}"
        )

    metrics = normalise_metrics(output.metrics)
    diagnostics = _normalise_diagnostics(output.diagnostics)
    names, arrays = _series_to_arrays(output.series)
    schema = pa.schema(
        [
            pa.field(name, array.type, nullable=False)
            for name, array in zip(names, arrays, strict=True)
        ]
    )
    table = pa.Table.from_arrays(arrays, schema=schema)
    metadata = {
        META_DIAGNOSTICS: canonical_json(diagnostics),
        META_ENGINE_VERSION: engine_version,
        META_METRICS: canonical_json(metrics),
        META_SCHEMA_VERSION: RESULT_SCHEMA_VERSION_STR,
        META_SPEC_HASH: spec_hash,
    }

    sink = pa.BufferOutputStream()
    writer = pq.ParquetWriter(sink, schema, **WRITER_OPTIONS)
    try:
        writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        writer.add_key_value_metadata(metadata)
    finally:
        writer.close()
    data = bytes(sink.getvalue().to_pybytes())
    return EncodedResult(data=data, row_count=int(table.num_rows), metrics=metrics)


def normalise_metrics(metrics: Mapping[str, Any]) -> dict[str, MetricValue]:
    """Validate metrics and return them with sorted keys and plain Python values.

    Each value must be ``None``, an ``int`` within int64 or a finite ``float``. numpy
    scalars are converted to Python numbers. Booleans, NaN and +-inf are rejected with
    ``ValueError``. ``-0.0`` becomes ``0.0``.
    """
    if not isinstance(metrics, Mapping):
        raise ValueError(f"metrics must be a mapping, got {type(metrics).__name__}")
    _check_names(metrics, "metric")
    return {name: _metric_value(name, metrics[name]) for name in sorted(metrics)}


def _metric_value(name: str, value: Any) -> MetricValue:
    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"metric {name!r} is a boolean; metrics must be numbers or None")
    if isinstance(value, np.integer):
        value = int(value)
    elif isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, int):
        if not _INT64_MIN <= value <= _INT64_MAX:
            raise ValueError(f"metric {name!r}={value} does not fit in int64")
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite or None, got {value!r}")
        return 0.0 if value == 0.0 else float(value)
    raise ValueError(f"metric {name!r} has unsupported type {type(value).__name__}")


def _normalise_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, DiagnosticValue]:
    if not isinstance(diagnostics, Mapping):
        raise ValueError(f"diagnostics must be a mapping, got {type(diagnostics).__name__}")
    out: dict[str, DiagnosticValue] = {}
    for key, value in diagnostics.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"diagnostic keys must be non-empty strings, got {key!r}")
        if isinstance(value, np.generic):
            value = value.item()
        if value is not None and not isinstance(value, bool | int | float | str):
            raise ValueError(
                f"diagnostic {key!r} has unsupported type {type(value).__name__} "
                "(expected str, int, float, bool or None)"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"diagnostic {key!r} must be finite, got {value!r}")
        out[key] = value
    return out


def _check_names(mapping: Mapping[Any, Any], what: str) -> None:
    for key in mapping:
        if not isinstance(key, str) or not _NAME_RE.fullmatch(key):
            raise ValueError(f"invalid {what} name {key!r}: must match {NAME_PATTERN}")


def _series_to_arrays(series: Mapping[str, Any]) -> tuple[list[str], list[pa.Array]]:
    if not isinstance(series, Mapping):
        raise ValueError(f"series must be a mapping, got {type(series).__name__}")
    _check_names(series, "series column")
    if TS_COLUMN not in series:
        raise ValueError(f"series must contain a {TS_COLUMN!r} column")
    names = [TS_COLUMN, *sorted(k for k in series if k != TS_COLUMN)]
    lengths: dict[str, int] = {}
    for name in names:
        values = series[name]
        if not isinstance(values, np.ndarray):
            raise ValueError(f"series {name!r} must be a numpy array, got {type(values).__name__}")
        if isinstance(values, np.ma.MaskedArray):  # the mask would be silently dropped
            raise ValueError(f"series {name!r} is a masked array; use NaN for missing values")
        if values.ndim != 1:
            raise ValueError(f"series {name!r} must be 1-D, got shape {values.shape}")
        lengths[name] = int(values.shape[0])
    if len(set(lengths.values())) > 1:
        raise ValueError(f"series arrays must have equal length, got {lengths}")
    arrays = [_ts_array(series[TS_COLUMN])]
    arrays.extend(_value_array(name, series[name]) for name in names[1:])
    return names, arrays


def _ts_array(values: npt.NDArray[Any]) -> pa.Array:
    if values.dtype.kind == "M":
        if np.datetime_data(values.dtype) != ("ns", 1):
            raise ValueError(f"series 'ts' must be datetime64[ns] or int64 ns, got {values.dtype}")
        if np.isnat(values).any():
            raise ValueError("series 'ts' contains NaT")
        # astype() first: normalises a non-native byte order before reinterpreting.
        ints = values.astype(np.dtype("datetime64[ns]"), copy=False).view(np.int64)
    else:
        ints = _as_int64("ts", values)
    return pa.array(np.ascontiguousarray(ints, dtype=np.int64), type=TS_TYPE)


def _value_array(name: str, values: npt.NDArray[Any]) -> pa.Array:
    kind = values.dtype.kind
    if kind == "f":
        if values.dtype.itemsize > 8:
            raise ValueError(f"series {name!r}: {values.dtype} does not fit in float64 exactly")
        floats = np.array(values, dtype=np.float64, copy=True)
        with np.errstate(invalid="ignore"):  # signalling NaNs must not warn here
            floats += 0.0  # folds -0.0 into +0.0 (IEEE round-to-nearest: -0 + +0 = +0)
        floats[np.isnan(floats)] = np.nan  # one canonical quiet-NaN bit pattern
        return pa.array(floats, type=pa.float64())
    if kind in "iu":
        return pa.array(np.ascontiguousarray(_as_int64(name, values)), type=pa.int64())
    if kind == "b":
        return pa.array(np.ascontiguousarray(values, dtype=np.bool_), type=pa.bool_())
    raise ValueError(
        f"series {name!r} has unsupported dtype {values.dtype} (use float, int or bool)"
    )


def _as_int64(name: str, values: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    kind = values.dtype.kind
    if kind not in "iu":
        raise ValueError(f"series {name!r} must have an integer dtype, got {values.dtype}")
    if (
        kind == "u"
        and values.dtype.itemsize >= 8
        and values.size
        and int(values.max()) > _INT64_MAX
    ):
        raise ValueError(f"series {name!r} has values that do not fit in int64")
    return values.astype(np.int64, copy=False)


# --------------------------------------------------------------------------------------
# Decoding and validation
# --------------------------------------------------------------------------------------


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON number {token}")


def _read_parquet(data: bytes) -> tuple[pa.Table, dict[str, str]]:
    """Parse artifact bytes. Raises ``ValueError`` if they are not a readable Parquet file
    with UTF-8 key-value metadata."""
    if not isinstance(data, bytes | bytearray | memoryview):
        raise ValueError(f"artifact data must be bytes, got {type(data).__name__}")
    try:
        parquet_file = pq.ParquetFile(pa.BufferReader(bytes(data)))
        table = parquet_file.read()
        raw = parquet_file.metadata.metadata or {}
    except Exception as exc:  # pyarrow raises ArrowInvalid/ArrowIOError/OSError subclasses
        raise ValueError(f"not a readable Parquet file: {exc}") from exc
    try:
        metadata = {bytes(k).decode("utf-8"): bytes(v).decode("utf-8") for k, v in raw.items()}
    except UnicodeDecodeError as exc:
        raise ValueError(f"artifact metadata is not valid UTF-8: {exc}") from exc
    return table, metadata


def _parse_json_object(text: str, what: str) -> dict[str, Any]:
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:  # JSONDecodeError is a ValueError
        raise ValueError(f"{what} metadata is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{what} metadata must be a JSON object, got {type(value).__name__}")
    return value


def _parse_diagnostics(text: str) -> dict[str, DiagnosticValue]:
    raw = _parse_json_object(text, "diagnostics")
    for key, value in raw.items():
        if value is not None and not isinstance(value, bool | int | float | str):
            raise ValueError(f"diagnostic {key!r} is not a scalar: {value!r}")
    return raw


def decode_result(data: bytes) -> DecodedResult:
    """Decode an artifact produced by :func:`encode_result`.

    Strict about structure: required metadata present, metrics finite-or-null numbers,
    diagnostics flat scalars, integer schema version. Raises ``ValueError`` otherwise.
    It does not reject other schema versions or a ``spec_hash`` the caller did not
    expect. :func:`validate_artifact` does those checks.
    """
    table, metadata = _read_parquet(data)
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"not a QuantGrid result artifact: missing metadata {missing}")
    spec_hash = metadata[META_SPEC_HASH]
    if not is_sha256_hex(spec_hash):
        raise ValueError(f"artifact spec_hash is not a SHA-256 hex digest: {spec_hash!r}")
    version_text = metadata[META_SCHEMA_VERSION]
    if not version_text.isdigit() or not version_text.isascii():
        raise ValueError(f"artifact result_schema_version is not an integer: {version_text!r}")
    metrics = normalise_metrics(_parse_json_object(metadata[META_METRICS], "metrics"))
    return DecodedResult(
        spec_hash=spec_hash,
        metrics=metrics,
        diagnostics=_parse_diagnostics(metadata[META_DIAGNOSTICS]),
        engine_version=metadata[META_ENGINE_VERSION],
        result_schema_version=int(version_text),
        table=table,
    )


def validate_artifact(
    data: bytes,
    *,
    expected_spec_hash: str,
    expected_metrics: Mapping[str, MetricValue] | None = None,
) -> list[str]:
    """Check that ``data`` is a well-formed v1 artifact for ``expected_spec_hash``.

    Returns a list of human-readable problems. An empty list means the artifact is OK.
    It never raises for bad *data*. It raises ``ValueError`` only for a malformed
    ``expected_spec_hash`` argument. If ``expected_metrics`` is given (for example
    ``result_artifacts.metrics`` read back from PostgreSQL), the embedded metrics must
    have the same keys and numerically equal values. They are compared by value, not
    text, because JSONB re-renders numbers: the float ``1e+16`` comes back as the
    integer ``10000000000000000``.
    """
    if not is_sha256_hex(expected_spec_hash):
        raise ValueError(f"expected_spec_hash must be a SHA-256 hex digest: {expected_spec_hash!r}")
    try:
        table, metadata = _read_parquet(data)
    except ValueError as exc:
        return [str(exc)]

    problems: list[str] = []
    for key in METADATA_KEYS:
        if key not in metadata:
            problems.append(f"missing metadata key {key!r}")
    unexpected = sorted(set(metadata) - set(METADATA_KEYS))
    if unexpected:
        problems.append(f"unexpected metadata keys {unexpected}")

    spec_hash = metadata.get(META_SPEC_HASH)
    if spec_hash is not None and spec_hash != expected_spec_hash:
        problems.append(
            f"spec_hash mismatch: artifact has {spec_hash!r}, expected {expected_spec_hash!r}"
        )
    version = metadata.get(META_SCHEMA_VERSION)
    if version is not None and version != RESULT_SCHEMA_VERSION_STR:
        problems.append(
            f"result_schema_version is {version!r}, expected {RESULT_SCHEMA_VERSION_STR!r}"
        )
    engine_version = metadata.get(META_ENGINE_VERSION)
    if engine_version is not None and not _ENGINE_VERSION_RE.fullmatch(engine_version):
        problems.append(f"invalid engine_version {engine_version!r}")

    metrics_text = metadata.get(META_METRICS)
    if metrics_text is not None:
        problems.extend(_metrics_problems(metrics_text, expected_metrics))
    diagnostics_text = metadata.get(META_DIAGNOSTICS)
    if diagnostics_text is not None:
        try:
            diagnostics = _parse_diagnostics(diagnostics_text)
            if canonical_json(diagnostics) != diagnostics_text:
                problems.append("diagnostics metadata is not canonical JSON")
        except (ValueError, TypeError) as exc:
            problems.append(f"invalid diagnostics metadata: {exc}")

    problems.extend(_schema_problems(table))
    return problems


def _metrics_problems(text: str, expected: Mapping[str, MetricValue] | None) -> list[str]:
    try:
        metrics = normalise_metrics(_parse_json_object(text, "metrics"))
    except ValueError as exc:
        return [f"invalid metrics metadata: {exc}"]
    problems: list[str] = []
    actual_text = canonical_json(metrics)
    if actual_text != text:
        problems.append("metrics metadata is not canonical JSON")
    if expected is not None:
        try:
            expected_metrics = normalise_metrics(expected)
        except ValueError as exc:
            return [*problems, f"expected_metrics are invalid: {exc}"]
        # Python compares int/float exactly by value; None only equals None.
        if expected_metrics != metrics:
            problems.append(
                f"metrics mismatch: artifact has {actual_text}, "
                f"expected {canonical_json(expected_metrics)}"
            )
    return problems


def _schema_problems(table: pa.Table) -> list[str]:
    problems: list[str] = []
    names = [str(n) for n in table.column_names]
    if not names or names[0] != TS_COLUMN:
        problems.append(f"first column must be {TS_COLUMN!r}, got {names[:1]}")
    else:
        ts_type = table.schema.field(TS_COLUMN).type
        if ts_type != TS_TYPE:
            problems.append(f"column 'ts' has type {ts_type}, expected {TS_TYPE}")
    rest = names[1:] if names[:1] == [TS_COLUMN] else names
    if rest != sorted(rest) or len(set(rest)) != len(rest):
        problems.append(f"value columns must be unique and sorted, got {rest}")
    allowed = (pa.float64(), pa.int64(), pa.bool_())
    for name in names:
        if not _NAME_RE.fullmatch(name):
            problems.append(f"invalid column name {name!r}")
        column = table.column(name)
        if name != TS_COLUMN and column.type not in allowed:
            problems.append(f"column {name!r} has unsupported type {column.type}")
        if column.null_count:
            problems.append(f"column {name!r} contains {column.null_count} null(s)")
    return problems
