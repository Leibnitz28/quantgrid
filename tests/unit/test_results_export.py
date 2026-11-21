"""Tests for the DuckDB result helpers (``quantgrid.results.export``)."""

from __future__ import annotations

import csv
import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest

from quantgrid.domain.types import TaskOutput
from quantgrid.results.codec import decode_result, encode_result, validate_artifact
from quantgrid.results.export import (
    MetricsExport,
    export_metrics,
    metrics_table,
    scan_artifacts,
    summarize_metrics,
)
from quantgrid.storage.artifacts import LocalArtifactStore


def _h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _row(
    ordinal: int,
    *,
    sharpe: float | None,
    task_id: str | uuid.UUID | None = None,
    params: dict[str, Any] | None = None,
    output_sha: str | None = "auto",
    **metrics: Any,
) -> dict[str, Any]:
    return {
        "task_id": task_id if task_id is not None else f"task-{ordinal:03d}",
        "ordinal": ordinal,
        "spec_hash": _h(f"spec-{ordinal}"),
        "output_sha": _h(f"out-{ordinal}") if output_sha == "auto" else output_sha,
        "params": params
        if params is not None
        else {"lookback": 10 + ordinal, "entry_z": 1.5 + ordinal / 2, "stop_z": None},
        "metrics": {"sharpe": sharpe, **metrics},
    }


ROWS = [
    _row(2, sharpe=0.5, total_return=0.02, max_drawdown=-0.10, n_trades=4),
    _row(0, sharpe=1.25, total_return=0.05, max_drawdown=-0.04, n_trades=7),
    _row(1, sharpe=None, total_return=None, max_drawdown=None, n_trades=0, output_sha=None),
    _row(3, sharpe=-0.75, total_return=-0.03, max_drawdown=-0.20, n_trades=9),
]
EXPECTED_COLUMNS = (
    "task_id",
    "ordinal",
    "spec_hash",
    "output_sha",
    "p_entry_z",
    "p_lookback",
    "p_stop_z",
    "m_max_drawdown",
    "m_n_trades",
    "m_sharpe",
    "m_total_return",
)


# --------------------------------------------------------------------------------------
# export_metrics
# --------------------------------------------------------------------------------------


def test_export_parquet(tmp_path: Path) -> None:
    out = tmp_path / "metrics.parquet"
    result = export_metrics(ROWS, out, "parquet")

    assert result == MetricsExport(path=out, fmt="parquet", row_count=4, columns=EXPECTED_COLUMNS)
    con = duckdb.connect()
    try:
        described = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(out)]).fetchall()
        rows = con.execute("SELECT * FROM read_parquet(?)", [str(out)]).fetchall()
    finally:
        con.close()
    assert [(name, kind) for name, kind, *_ in described] == [
        ("task_id", "VARCHAR"),
        ("ordinal", "BIGINT"),
        ("spec_hash", "VARCHAR"),
        ("output_sha", "VARCHAR"),
        ("p_entry_z", "DOUBLE"),
        ("p_lookback", "BIGINT"),
        ("p_stop_z", "VARCHAR"),  # no value in any row
        ("m_max_drawdown", "DOUBLE"),
        ("m_n_trades", "BIGINT"),
        ("m_sharpe", "DOUBLE"),
        ("m_total_return", "DOUBLE"),
    ]
    assert [r[1] for r in rows] == [0, 1, 2, 3]  # ordered by ordinal
    assert rows[0] == (
        "task-000",
        0,
        _h("spec-0"),
        _h("out-0"),
        1.5,
        10,
        None,
        -0.04,
        7,
        1.25,
        0.05,
    )
    assert rows[1][3] is None  # task without a published output
    assert rows[1][9] is None  # undefined sharpe stays NULL


def test_export_csv(tmp_path: Path) -> None:
    out = tmp_path / "metrics.csv"
    export_metrics(ROWS, out, "csv")
    lines = out.read_text().splitlines()
    assert lines[0] == ",".join(EXPECTED_COLUMNS)
    assert lines[1] == f"task-000,0,{_h('spec-0')},{_h('out-0')},1.5,10,,-0.04,7,1.25,0.05"
    assert lines[2] == f"task-001,1,{_h('spec-1')},,2.0,11,,,0,,"
    with out.open(newline="") as fh:
        parsed = list(csv.DictReader(fh))
    assert [r["ordinal"] for r in parsed] == ["0", "1", "2", "3"]


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_export_is_deterministic(tmp_path: Path, fmt: str) -> None:
    a = export_metrics(ROWS, tmp_path / f"a.{fmt}", fmt)  # type: ignore[arg-type]
    b = export_metrics(list(reversed(ROWS)), tmp_path / f"b.{fmt}", fmt)  # type: ignore[arg-type]
    shuffled = [ROWS[i] for i in (3, 1, 0, 2)]
    c = export_metrics(shuffled, tmp_path / f"c.{fmt}", fmt)  # type: ignore[arg-type]
    assert a.columns == b.columns == c.columns == EXPECTED_COLUMNS
    assert a.path.read_bytes() == b.path.read_bytes() == c.path.read_bytes()


def test_export_overwrites_atomically_and_leaves_no_temp_files(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "dir" / "metrics.parquet"
    export_metrics(ROWS[:1], out, "parquet")
    export_metrics(ROWS, out, "parquet")
    assert pq.read_table(out).num_rows == 4
    assert sorted(p.name for p in out.parent.iterdir()) == ["metrics.parquet"]


def test_export_csv_ignores_compression_suffix(tmp_path: Path) -> None:
    out = tmp_path / "metrics.csv.gz"
    export_metrics(ROWS, out, "csv")
    assert out.read_text().startswith("task_id,ordinal,")


def test_export_empty(tmp_path: Path) -> None:
    csv_result = export_metrics([], tmp_path / "empty.csv", "csv")
    assert csv_result.row_count == 0
    assert csv_result.columns == ("task_id", "ordinal", "spec_hash", "output_sha")
    assert (tmp_path / "empty.csv").read_text().splitlines() == [
        "task_id,ordinal,spec_hash,output_sha"
    ]
    export_metrics([], tmp_path / "empty.parquet", "parquet")
    table = pq.read_table(tmp_path / "empty.parquet")
    assert table.num_rows == 0
    assert table.column_names == ["task_id", "ordinal", "spec_hash", "output_sha"]


def test_param_and_metric_type_unification() -> None:
    rows = [
        _row(0, sharpe=1, params={"a": 1, "b": 1, "c": True, "d": "x", "e": True, "f": None}),
        _row(1, sharpe=2.5, params={"a": 2, "b": 1.5, "c": False, "d": 3, "e": 1, "f": None}),
        _row(2, sharpe=None, params={"a": None, "b": None, "c": None, "d": 2.5, "e": None}),
    ]
    table = metrics_table(rows)
    types = {name: str(table.schema.field(name).type) for name in table.column_names}
    assert types["p_a"] == "int64"
    assert types["p_b"] == "double"  # int + float -> double
    assert types["p_c"] == "bool"
    assert types["p_d"] == "string"  # str + numbers -> text
    assert types["p_e"] == "string"  # bool + int -> text (never coerce bool to number)
    assert types["p_f"] == "string"  # no values at all
    assert types["m_sharpe"] == "double"
    assert table.column("p_b").to_pylist() == [1.0, 1.5, None]
    assert table.column("p_d").to_pylist() == ["x", "3", "2.5"]
    assert table.column("p_e").to_pylist() == ["true", "1", None]

    ints_only = metrics_table(
        [_row(0, sharpe=None, n_trades=3), _row(1, sharpe=None, n_trades=None)]
    )
    assert str(ints_only.schema.field("m_n_trades").type) == "int64"
    assert str(ints_only.schema.field("m_sharpe").type) == "double"  # all None -> double


def test_unusual_param_names_are_quoted_safely(tmp_path: Path) -> None:
    rows = [_row(0, sharpe=1.0, params={'weird, "name"': 1.0, "x'; DROP TABLE t; --": 2})]
    out = tmp_path / "weird.csv"
    result = export_metrics(rows, out, "csv")
    assert 'p_weird, "name"' in result.columns
    with out.open(newline="") as fh:
        header = next(csv.reader(fh))
    assert header == list(result.columns)


def test_uuid_task_ids_are_exported_as_strings(tmp_path: Path) -> None:
    task_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    table = metrics_table([_row(0, sharpe=1.0, task_id=task_id)])
    assert table.column("task_id").to_pylist() == [str(task_id)]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.pop("spec_hash"), "missing required keys"),
        (lambda r: r.update(task_id=""), "task_id"),
        (lambda r: r.update(task_id=7), "task_id"),
        (lambda r: r.update(ordinal=True), "ordinal"),
        (lambda r: r.update(ordinal=-1), "ordinal"),
        (lambda r: r.update(ordinal="3"), "ordinal"),
        (lambda r: r.update(spec_hash="abc"), "spec_hash"),
        (lambda r: r.update(output_sha="ABC"), "output_sha"),
        (lambda r: r.update(metrics={"sharpe": float("nan")}), "finite"),
        (lambda r: r.update(metrics={"sharpe": float("inf")}), "finite"),
        (lambda r: r.update(metrics={"sharpe": True}), "number"),
        (lambda r: r.update(metrics={"sharpe": "1.0"}), "number"),
        (lambda r: r.update(metrics={"n": 2**64}), "int64"),
        (lambda r: r.update(metrics=[1.0]), "mapping"),
        (lambda r: r.update(params={"": 1}), "non-empty strings"),
        (lambda r: r.update(params={"a": [1, 2]}), "scalar"),
    ],
)
def test_invalid_rows_are_rejected(mutate: Any, match: str, tmp_path: Path) -> None:
    row = _row(0, sharpe=1.0)
    mutate(row)
    with pytest.raises(ValueError, match=match):
        export_metrics([row], tmp_path / "x.csv", "csv")
    with pytest.raises(ValueError, match=match):
        summarize_metrics([row])
    assert not (tmp_path / "x.csv").exists()


def test_duplicate_task_ids_are_rejected(tmp_path: Path) -> None:
    rows = [_row(0, sharpe=1.0), _row(1, sharpe=2.0, task_id="task-000")]
    with pytest.raises(ValueError, match="duplicate task_id"):
        export_metrics(rows, tmp_path / "x.parquet", "parquet")


def test_invalid_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fmt"):
        export_metrics(ROWS, tmp_path / "x.json", "json")  # type: ignore[arg-type]


def test_extra_row_keys_are_ignored() -> None:
    rows = [{**r, "state": "SUCCEEDED", "attempt": 1} for r in ROWS]
    assert metrics_table(rows).column_names == list(EXPECTED_COLUMNS)


# --------------------------------------------------------------------------------------
# scan_artifacts
# --------------------------------------------------------------------------------------


def _artifact(path: Path, spec_text: str, n: int) -> tuple[Path, str]:
    spec_hash = _h(spec_text)
    output = TaskOutput(
        metrics={"sharpe": 1.0},
        series={"ts": np.arange(n, dtype=np.int64), "pnl": np.linspace(0.0, 1.0, n)},
    )
    path.write_bytes(encode_result(spec_hash, output, engine_version="test@1").data)
    return path, spec_hash


def test_scan_artifacts(tmp_path: Path) -> None:
    a, hash_a = _artifact(tmp_path / "a.parquet", "spec-a", 10)
    b, hash_b = _artifact(tmp_path / "b.parquet", "spec-b", 25)
    scans = scan_artifacts([b, a, b])
    assert scans == [
        {"path": b, "row_count": 25, "spec_hash": hash_b, "error": None},
        {"path": a, "row_count": 10, "spec_hash": hash_a, "error": None},
        {"path": b, "row_count": 25, "spec_hash": hash_b, "error": None},
    ]
    assert scan_artifacts([]) == []


def test_scan_isolates_missing_and_corrupt_files(tmp_path: Path) -> None:
    good, good_hash = _artifact(tmp_path / "good.parquet", "spec-good", 5)
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(good.read_bytes()[:-40])  # truncated footer
    garbage = tmp_path / "garbage.parquet"
    garbage.write_bytes(b"definitely not parquet")
    missing = tmp_path / "missing.parquet"

    scans = scan_artifacts([good, corrupt, missing, garbage])

    assert scans[0] == {"path": good, "row_count": 5, "spec_hash": good_hash, "error": None}
    for scan, path in zip(scans[1:], [corrupt, missing, garbage], strict=True):
        assert scan["path"] == path
        assert scan["row_count"] is None
        assert scan["spec_hash"] is None
        assert scan["error"]
    assert scans[2]["error"] == "file not found"


def test_scan_plain_parquet_without_quantgrid_metadata(tmp_path: Path) -> None:
    plain = tmp_path / "plain.parquet"
    export_metrics(ROWS, plain, "parquet")
    assert scan_artifacts([plain]) == [
        {"path": plain, "row_count": 4, "spec_hash": None, "error": None}
    ]


def test_scan_paths_with_glob_characters_are_not_expanded(tmp_path: Path) -> None:
    # DuckDB would read "a[1].parquet" as a glob matching "a1.parquet"; the scan must
    # report the literal file.
    literal, literal_hash = _artifact(tmp_path / "a[1].parquet", "spec-literal", 3)
    _artifact(tmp_path / "a1.parquet", "spec-decoy", 99)
    star, star_hash = _artifact(tmp_path / "s*.parquet", "spec-star", 4)
    _artifact(tmp_path / "sx.parquet", "spec-decoy-2", 98)

    scans = scan_artifacts([literal, star])

    assert scans == [
        {"path": literal, "row_count": 3, "spec_hash": literal_hash, "error": None},
        {"path": star, "row_count": 4, "spec_hash": star_hash, "error": None},
    ]


def test_scan_many_artifacts_in_one_query(tmp_path: Path) -> None:
    template, _ = _artifact(tmp_path / "template.parquet", "spec-template", 7)
    paths = []
    for i in range(50):
        path = tmp_path / f"copy-{i:02d}.parquet"
        shutil.copyfile(template, path)
        paths.append(path)
    scans = scan_artifacts(paths)
    assert [s["path"] for s in scans] == paths
    assert {s["row_count"] for s in scans} == {7}
    assert {s["spec_hash"] for s in scans} == {_h("spec-template")}


# --------------------------------------------------------------------------------------
# summarize_metrics
# --------------------------------------------------------------------------------------


def test_summarize_metrics() -> None:
    summary = summarize_metrics(ROWS)
    values = np.array([1.25, 0.5, -0.75])
    assert summary["key"] == "sharpe"
    assert summary["count"] == 4
    assert summary["n_with_value"] == 3
    assert (summary["best_task_id"], summary["best_ordinal"], summary["best_value"]) == (
        "task-000",
        0,
        1.25,
    )
    assert (summary["worst_task_id"], summary["worst_ordinal"], summary["worst_value"]) == (
        "task-003",
        3,
        -0.75,
    )
    assert summary["mean"] == pytest.approx(float(np.mean(values)), abs=1e-15)
    assert summary["p50"] == pytest.approx(float(np.percentile(values, 50)), abs=1e-15)
    assert summary["p95"] == pytest.approx(float(np.percentile(values, 95)), abs=1e-15)
    assert summary["mean_total_return"] == pytest.approx((0.02 + 0.05 - 0.03) / 3)
    assert summary["mean_max_drawdown"] == pytest.approx((-0.10 - 0.04 - 0.20) / 3)
    assert summary["mean_n_trades"] == pytest.approx((4 + 7 + 0 + 9) / 4)


def test_summary_percentiles_match_numpy_linear_interpolation() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=101)
    rows = [_row(i, sharpe=float(v)) for i, v in enumerate(values)]
    summary = summarize_metrics(rows)
    assert summary["p50"] == pytest.approx(float(np.percentile(values, 50)), rel=1e-12)
    assert summary["p95"] == pytest.approx(float(np.percentile(values, 95)), rel=1e-12)
    assert summary["mean"] == pytest.approx(float(np.mean(values)), rel=1e-12, abs=1e-15)


def test_summary_ties_are_broken_by_ordinal() -> None:
    rows = [
        _row(5, sharpe=2.0, task_id="a-late"),
        _row(1, sharpe=2.0, task_id="z-early"),
        _row(3, sharpe=-1.0, task_id="b"),
        _row(2, sharpe=-1.0, task_id="c"),
    ]
    for ordering in (rows, list(reversed(rows))):
        summary = summarize_metrics(ordering)
        assert summary["best_task_id"] == "z-early"
        assert summary["worst_task_id"] == "c"


def test_summary_is_independent_of_row_order() -> None:
    rng = np.random.default_rng(11)
    rows = [
        _row(i, sharpe=float(v), total_return=float(v) / 10)
        for i, v in enumerate(rng.normal(size=40))
    ]
    assert summarize_metrics(rows) == summarize_metrics(list(reversed(rows)))


def test_summary_lower_is_better_and_other_keys() -> None:
    summary = summarize_metrics(ROWS, key="max_drawdown", higher_is_better=False)
    assert summary["key"] == "max_drawdown"
    assert summary["best_task_id"] == "task-003"  # lowest value wins
    assert summary["worst_task_id"] == "task-000"


def test_summary_without_values() -> None:
    empty = summarize_metrics([])
    assert empty["count"] == 0
    assert empty["n_with_value"] == 0
    assert all(
        empty[k] is None
        for k in (
            "best_task_id",
            "best_value",
            "worst_task_id",
            "mean",
            "p50",
            "p95",
            "mean_n_trades",
        )
    )
    no_sharpe = summarize_metrics([_row(0, sharpe=None), _row(1, sharpe=None)])
    assert no_sharpe["count"] == 2
    assert no_sharpe["n_with_value"] == 0
    assert no_sharpe["best_task_id"] is None
    assert no_sharpe["mean"] is None
    missing_key = summarize_metrics(ROWS, key="does_not_exist")
    assert missing_key["n_with_value"] == 0


# --------------------------------------------------------------------------------------
# End to end: codec -> artifact store -> scan/validate -> export/summary
# --------------------------------------------------------------------------------------


def test_publish_scan_export_and_summarize(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    rows: list[dict[str, Any]] = []
    for ordinal in range(6):
        n = 50 + ordinal
        spec_hash = _h(f"e2e-spec-{ordinal}")
        output = TaskOutput(
            metrics={"sharpe": ordinal / 4 - 0.5, "n_trades": ordinal, "total_return": None},
            series={"ts": np.arange(n, dtype=np.int64), "pnl": np.linspace(-1.0, 1.0, n)},
        )
        encoded = encode_result(spec_hash, output, engine_version="e2e@1")
        published = store.commit(store.stage(encoded.data, hint=f"task-{ordinal}"))
        # A retry of the same task produces the same bytes and therefore the same artifact.
        retry = encode_result(spec_hash, output, engine_version="e2e@1")
        assert store.commit(store.stage(retry.data, hint=f"task-{ordinal}-retry")).created is False
        assert published.sha256 == encoded.sha256
        rows.append(
            {
                "task_id": f"task-{ordinal}",
                "ordinal": ordinal,
                "spec_hash": spec_hash,
                "output_sha": published.sha256,
                "params": {"lookback": 10 * (ordinal + 1)},
                "metrics": encoded.metrics,
                "output_uri": published.uri,
            }
        )

    listed = list(store.iter_artifacts())
    assert len(listed) == len(rows)  # exactly one artifact per task, despite the retries
    by_sha = {row["output_sha"]: row for row in rows}
    scans = scan_artifacts([path for _, path in listed])
    for (uri, _), scan in zip(listed, scans, strict=True):
        row = by_sha[store.path_for_uri(uri).stem]
        assert scan["error"] is None
        assert scan["spec_hash"] == row["spec_hash"]
        assert scan["row_count"] == 50 + row["ordinal"]
        data = store.read_bytes(uri)
        assert (
            validate_artifact(
                data, expected_spec_hash=row["spec_hash"], expected_metrics=row["metrics"]
            )
            == []
        )
        assert decode_result(data).metrics == row["metrics"]

    exported = export_metrics(rows, tmp_path / "batch.parquet", "parquet")
    assert exported.columns == (
        "task_id",
        "ordinal",
        "spec_hash",
        "output_sha",
        "p_lookback",
        "m_n_trades",
        "m_sharpe",
        "m_total_return",
    )
    summary = summarize_metrics(rows)
    assert (summary["best_task_id"], summary["worst_task_id"]) == ("task-5", "task-0")
    assert summary["mean_total_return"] is None
    assert summary["mean_n_trades"] == pytest.approx(2.5)
