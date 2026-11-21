"""DuckDB helpers for merged result tables, used by the results endpoint, CLI export and verifier.

Each metrics row (:class:`MetricsRow`) needs ``task_id`` (unique), ``ordinal`` and
``spec_hash``; ``output_sha``, ``params`` and ``metrics`` are optional and other keys are
ignored. Invalid rows raise ``ValueError``. Output is deterministic: rows ordered by
``(ordinal, task_id)``, columns ``task_id, ordinal, spec_hash, output_sha``, then sorted
``p_<param>`` and ``m_<metric>``, so the same rows in any order export identical bytes.
"""

from __future__ import annotations

import contextlib
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NotRequired, TypedDict

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from quantgrid.domain.hashing import is_sha256_hex
from quantgrid.domain.specs import ParamValue
from quantgrid.domain.types import MetricValue

ExportFormat = Literal["parquet", "csv"]
FIXED_COLUMNS: Final = ("task_id", "ordinal", "spec_hash", "output_sha")
PARAM_PREFIX: Final = "p_"
METRIC_PREFIX: Final = "m_"
SPEC_HASH_METADATA_KEY: Final = b"quantgrid.spec_hash"
#: Metrics whose means :func:`summarize_metrics` always reports.
SUMMARY_MEAN_METRICS: Final = ("total_return", "max_drawdown", "n_trades")
#: DuckDB's table functions expand these characters as globs, so a path containing one
#: could match other files. Such paths are read with pyarrow instead.
_GLOB_CHARS: Final = frozenset("*?[")
_INT64_MIN: Final = -(2**63)
_INT64_MAX: Final = 2**63 - 1


class MetricsRow(TypedDict):
    task_id: str | uuid.UUID
    ordinal: int
    spec_hash: str
    output_sha: NotRequired[str | None]
    params: NotRequired[Mapping[str, ParamValue] | None]
    metrics: NotRequired[Mapping[str, MetricValue] | None]


class ArtifactScan(TypedDict):
    """One :func:`scan_artifacts` entry; ``error`` is set when the file is missing or unreadable."""

    path: Path
    row_count: int | None
    spec_hash: str | None
    error: str | None


class MetricsSummary(TypedDict):
    key: str
    count: int
    n_with_value: int
    best_task_id: str | None
    best_ordinal: int | None
    best_value: float | None
    worst_task_id: str | None
    worst_ordinal: int | None
    worst_value: float | None
    mean: float | None
    p50: float | None
    p95: float | None
    mean_total_return: float | None
    mean_max_drawdown: float | None
    mean_n_trades: float | None


@dataclass(frozen=True, slots=True)
class MetricsExport:
    path: Path
    fmt: ExportFormat
    row_count: int
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Row:
    task_id: str
    ordinal: int
    spec_hash: str
    output_sha: str | None
    params: dict[str, ParamValue]
    metrics: dict[str, MetricValue]


def _row_error(index: int, message: str) -> ValueError:
    return ValueError(f"metrics row {index}: {message}")


def _metric_value(index: int, name: str, value: Any) -> MetricValue:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _row_error(index, f"metric {name!r} must be a number or None, got {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise _row_error(index, f"metric {name!r} must be finite or None, got {value!r}")
    if isinstance(value, int) and not _INT64_MIN <= value <= _INT64_MAX:
        raise _row_error(index, f"metric {name!r}={value} does not fit in int64")
    return value


def _named_mapping(index: int, raw: Any, what: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise _row_error(index, f"{what} must be a mapping or None, got {type(raw).__name__}")
    for name in raw:
        if not isinstance(name, str) or not name:
            raise _row_error(index, f"{what} names must be non-empty strings, got {name!r}")
    return dict(raw)


def _normalise_rows(rows: Sequence[Mapping[str, Any]]) -> list[_Row]:
    out: list[_Row] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise _row_error(index, f"must be a mapping, got {type(row).__name__}")
        missing = [k for k in ("task_id", "ordinal", "spec_hash") if k not in row]
        if missing:
            raise _row_error(index, f"missing required keys {missing}")
        raw_id = row["task_id"]
        if isinstance(raw_id, uuid.UUID):
            task_id = str(raw_id)
        elif isinstance(raw_id, str) and raw_id:
            task_id = raw_id
        else:
            raise _row_error(index, f"task_id must be a UUID or non-empty string, got {raw_id!r}")
        if task_id in seen:
            raise _row_error(
                index, f"duplicate task_id {task_id!r} (also row {seen[task_id]}): one row per task"
            )
        seen[task_id] = index
        ordinal = row["ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise _row_error(index, f"ordinal must be an int >= 0, got {ordinal!r}")
        spec_hash = row["spec_hash"]
        if not is_sha256_hex(spec_hash):
            raise _row_error(index, f"spec_hash is not a SHA-256 hex digest: {spec_hash!r}")
        output_sha = row.get("output_sha")
        if output_sha is not None and not is_sha256_hex(output_sha):
            raise _row_error(index, f"output_sha is not a SHA-256 hex digest: {output_sha!r}")
        params = _named_mapping(index, row.get("params"), "params")
        for name, value in params.items():
            if value is not None and not isinstance(value, bool | int | float | str):
                raise _row_error(index, f"param {name!r} must be a scalar or None, got {value!r}")
        metrics = {
            name: _metric_value(index, name, value)
            for name, value in _named_mapping(index, row.get("metrics"), "metrics").items()
        }
        out.append(_Row(task_id, ordinal, spec_hash, output_sha, params, metrics))
    out.sort(key=lambda r: (r.ordinal, r.task_id))
    return out


def _value_kind(value: ParamValue) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int" if _INT64_MIN <= value <= _INT64_MAX else "text"
    if isinstance(value, float):
        return "float"
    return "text"


def _as_text(value: ParamValue) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)  # shortest round-trip representation
    return str(value)


def _param_array(values: list[ParamValue]) -> pa.Array:
    """Arrow column of param values: bool, int64, float64 (ints mixed with floats), else text."""
    kinds = {_value_kind(v) for v in values if v is not None}
    if kinds == {"bool"}:
        return pa.array(values, type=pa.bool_())
    if kinds == {"int"}:
        return pa.array(values, type=pa.int64())
    if kinds and kinds <= {"int", "float"}:
        return pa.array([None if v is None else float(v) for v in values], type=pa.float64())
    return pa.array([_as_text(v) for v in values], type=pa.string())


def _metric_array(values: list[MetricValue]) -> pa.Array:
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, int) for v in present):
        return pa.array(values, type=pa.int64())
    return pa.array([None if v is None else float(v) for v in values], type=pa.float64())


def metrics_table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    normalised = _normalise_rows(rows)
    param_names = sorted({name for r in normalised for name in r.params})
    metric_names = sorted({name for r in normalised for name in r.metrics})
    names: list[str] = list(FIXED_COLUMNS)
    arrays: list[pa.Array] = [
        pa.array([r.task_id for r in normalised], type=pa.string()),
        pa.array([r.ordinal for r in normalised], type=pa.int64()),
        pa.array([r.spec_hash for r in normalised], type=pa.string()),
        pa.array([r.output_sha for r in normalised], type=pa.string()),
    ]
    for name in param_names:
        names.append(f"{PARAM_PREFIX}{name}")
        arrays.append(_param_array([r.params.get(name) for r in normalised]))
    for name in metric_names:
        names.append(f"{METRIC_PREFIX}{name}")
        arrays.append(_metric_array([r.metrics.get(name) for r in normalised]))
    return pa.Table.from_arrays(arrays, names=names)


def export_metrics(
    rows: Sequence[Mapping[str, Any]], out_path: Path, fmt: ExportFormat
) -> MetricsExport:
    """Write the merged metrics table to ``out_path`` as zstd Parquet or uncompressed CSV.

    The file is written to a temp name and renamed, so readers never see a partial export.
    """
    if fmt not in ("parquet", "csv"):
        raise ValueError(f"fmt must be 'parquet' or 'csv', got {fmt!r}")
    table = metrics_table(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{uuid.uuid4().hex}.tmp")
    con = duckdb.connect(":memory:")
    try:
        # One thread + insertion order => identical bytes for identical input.
        con.execute("SET threads TO 1")
        con.execute("SET preserve_insertion_order = true")
        relation = con.from_arrow(table).order("ordinal ASC, task_id ASC")
        if fmt == "parquet":
            relation.write_parquet(str(tmp_path), compression="zstd")
        else:
            relation.write_csv(str(tmp_path), header=True, sep=",", compression="none")
        os.replace(tmp_path, out_path)
    finally:
        con.close()
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
    return MetricsExport(
        path=out_path,
        fmt=fmt,
        row_count=int(table.num_rows),
        columns=tuple(str(n) for n in table.column_names),
    )


def _scan_entry(
    path: Path, row_count: int | None, spec_hash: str | None, error: str | None = None
) -> ArtifactScan:
    return {"path": path, "row_count": row_count, "spec_hash": spec_hash, "error": error}


def _decode_spec_hashes(values: list[bytes]) -> tuple[str | None, str | None]:
    distinct = sorted(set(values))
    if not distinct:
        return None, None
    if len(distinct) > 1:
        return None, "conflicting quantgrid.spec_hash metadata entries"
    return distinct[0].decode("utf-8", errors="replace"), None


def _scan_with_duckdb(keys: list[str]) -> dict[str, ArtifactScan]:
    """Scan non-glob paths in one DuckDB metadata query.

    Raises ``duckdb.Error`` for an unreadable file, ``RuntimeError`` if names cannot be matched.
    """
    con = duckdb.connect(":memory:")
    try:
        counts = con.execute(
            "SELECT file_name, num_rows FROM parquet_file_metadata(?)", [keys]
        ).fetchall()
        kv_rows = con.execute(
            "SELECT file_name, key, value FROM parquet_kv_metadata(?)", [keys]
        ).fetchall()
    finally:
        con.close()
    reported = {str(name) for name, _ in counts}
    if reported == set(keys) and len(counts) == len(keys):
        key_for = {key: key for key in keys}
    elif len(keys) == 1 and len(counts) == 1:
        key_for = {str(counts[0][0]): keys[0]}  # DuckDB spelled the one path differently
    else:
        raise RuntimeError("DuckDB returned unexpected file names")
    row_counts = {key_for[str(name)]: int(n) for name, n in counts}
    hashes: dict[str, list[bytes]] = {key: [] for key in keys}
    for name, key, value in kv_rows:
        target = key_for.get(str(name))
        if target is not None and bytes(key) == SPEC_HASH_METADATA_KEY:
            hashes[target].append(bytes(value))
    out: dict[str, ArtifactScan] = {}
    for key in keys:
        spec_hash, error = _decode_spec_hashes(hashes[key])
        out[key] = _scan_entry(Path(key), row_counts[key], spec_hash, error)
    return out


def _scan_with_pyarrow(key: str) -> ArtifactScan:
    path = Path(key)
    try:
        metadata = pq.ParquetFile(path).metadata
        num_rows = int(metadata.num_rows)
        kv = metadata.metadata or {}
    except Exception as exc:  # pyarrow raises ArrowInvalid/OSError subclasses
        return _scan_entry(path, None, None, f"unreadable parquet: {exc}")
    values = [bytes(v) for k, v in kv.items() if bytes(k) == SPEC_HASH_METADATA_KEY]
    spec_hash, error = _decode_spec_hashes(values)
    return _scan_entry(path, num_rows, spec_hash, error)


def scan_artifacts(paths: Sequence[Path]) -> list[ArtifactScan]:
    """Row count and embedded ``spec_hash`` of each Parquet file, in input order.

    If the query fails, files are rescanned one by one so a bad file only sets its own ``error``.
    """
    keys = [str(Path(p)) for p in paths]
    results: dict[str, ArtifactScan] = {}
    for_duckdb: list[str] = []
    for key in dict.fromkeys(keys):
        if not Path(key).is_file():
            results[key] = _scan_entry(Path(key), None, None, "file not found")
        elif _GLOB_CHARS.intersection(key):
            results[key] = _scan_with_pyarrow(key)
        else:
            for_duckdb.append(key)
    if for_duckdb:
        try:
            results.update(_scan_with_duckdb(for_duckdb))
        except (duckdb.Error, RuntimeError):
            for key in for_duckdb:
                try:
                    results.update(_scan_with_duckdb([key]))
                except (duckdb.Error, RuntimeError) as exc:
                    results[key] = _scan_entry(Path(key), None, None, f"unreadable parquet: {exc}")
    return [results[key] for key in keys]


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    h = (len(sorted_values) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def summarize_metrics(
    rows: Sequence[Mapping[str, Any]], *, key: str = "sharpe", higher_is_better: bool = True
) -> MetricsSummary:
    """Summarise one metric across tasks: count, best and worst task, mean, p50, p95.

    Rows without a value are ignored. Ties go to the lowest ordinal, then the lowest task_id.
    """
    normalised = _normalise_rows(rows)
    scored = [
        (float(value), r.ordinal, r.task_id)
        for r in normalised
        if (value := r.metrics.get(key)) is not None
    ]
    values = sorted(v for v, _, _ in scored)
    best = worst = None
    if scored:
        high = min(scored, key=lambda s: (-s[0], s[1], s[2]))
        low = min(scored, key=lambda s: (s[0], s[1], s[2]))
        best, worst = (high, low) if higher_is_better else (low, high)
    means: dict[str, float | None] = {}
    for name in SUMMARY_MEAN_METRICS:
        present = [float(v) for r in normalised if (v := r.metrics.get(name)) is not None]
        means[name] = _mean(present)
    return {
        "key": key,
        "count": len(normalised),
        "n_with_value": len(scored),
        "best_task_id": best[2] if best else None,
        "best_ordinal": best[1] if best else None,
        "best_value": best[0] if best else None,
        "worst_task_id": worst[2] if worst else None,
        "worst_ordinal": worst[1] if worst else None,
        "worst_value": worst[0] if worst else None,
        "mean": _mean(values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "mean_total_return": means["total_return"],
        "mean_max_drawdown": means["max_drawdown"],
        "mean_n_trades": means["n_trades"],
    }
