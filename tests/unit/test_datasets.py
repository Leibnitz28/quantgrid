"""Recorded synthetic datasets: generator, signatures, validation, Parquet I/O, catalog, cache."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quantgrid.domain import datasets as ds
from quantgrid.domain.datasets import (
    BAR_COLUMNS,
    DATASET_DEFINITIONS,
    DATASET_LICENSE,
    BarsCache,
    Catalog,
    CointegratedOU,
    IndependentRandomWalk,
    dataset_content_sha256,
    dataset_path,
    dataset_ref,
    generate_arrays,
    generate_bars,
    get_catalog,
    load_bars,
    materialize,
    validate_bars,
)
from quantgrid.domain.errors import DataIntegrityError
from quantgrid.domain.hashing import canonical_json
from quantgrid.domain.policy import PolicyService, QuotaPolicy
from quantgrid.domain.specs import DatasetRef, ExperimentBatchSpec, StrategyRef
from quantgrid.domain.types import Bars

ALL_DATASETS = sorted(DATASET_DEFINITIONS)
SMALL = "pairs-ou-small-v1"


@pytest.fixture(scope="module")
def generated() -> dict[str, Bars]:
    return {name: generate_bars(name) for name in ALL_DATASETS}


def _explicit_generate(defn: ds.DatasetDefinition) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Independent re-implementation of the documented generator (explicit recursions)."""
    n = defn.n_grid
    dt = defn.bar_interval_s / (365.25 * 86400)
    sq = math.sqrt(dt)
    rng = np.random.default_rng(defn.seed)
    ex = rng.standard_normal(n).tolist()
    e2 = rng.standard_normal(n).tolist()
    lx = [math.log(100.0)]
    for t in range(1, n):
        lx.append(lx[-1] + (defn.mu_x * dt + defn.sigma_x * sq * ex[t]))
    proc = defn.y_process
    if isinstance(proc, CointegratedOU):
        theta = math.log(2.0) / (proc.half_life_s / (365.25 * 86400))
        s = [0.0]
        for t in range(1, n):
            s.append(s[-1] - theta * s[-1] * dt + proc.sigma_s * sq * e2[t])
        ly = [proc.alpha_true + proc.beta_true * lx[t] + s[t] for t in range(n)]
    else:
        ly = [math.log(100.0) + proc.y_log_offset]
        for t in range(1, n):
            ly.append(ly[-1] + (proc.mu_y * dt + proc.sigma_y * sq * e2[t]))
    close_x = np.round(np.array([math.exp(v) for v in lx]), 6)
    close_y = np.round(np.array([math.exp(v) for v in ly]), 6)
    start_ns = int(defn.start.timestamp()) * 1_000_000_000
    ts = [start_ns + k * defn.bar_interval_s * 1_000_000_000 for k in range(n)]
    if defn.gap_fraction > 0:
        u = np.random.default_rng(defn.seed + 1).random(n)
        keep = [k == 0 or k == n - 1 or u[k] >= defn.gap_fraction for k in range(n)]
        ts = [t for t, k in zip(ts, keep, strict=True) if k]
        close_x = close_x[np.array(keep)]
        close_y = close_y[np.array(keep)]
    return ts, close_x, close_y


def _ar1(series: np.ndarray) -> float:
    a = series[1:] - series[1:].mean()
    b = series[:-1] - series[:-1].mean()
    return float((a * b).sum() / (b * b).sum())


def _write_table(
    path: Path, ts: np.ndarray, cx: np.ndarray, cy: np.ndarray, **kwargs: object
) -> None:
    table = pa.table(
        {
            "ts": pa.array(ts, type=pa.timestamp("ns", tz="UTC")),
            "close_x": pa.array(cx, type=pa.float64()),
            "close_y": pa.array(cy, type=pa.float64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Registry and generator
# --------------------------------------------------------------------------------------


class TestRegistry:
    def test_required_datasets_and_shapes(self) -> None:
        defs = DATASET_DEFINITIONS
        small = defs["pairs-ou-small-v1"]
        assert (small.n_grid, small.bar_interval_s) == (2_000, 86_400)
        assert small.start == datetime(2015, 1, 1, tzinfo=UTC)
        assert (defs["pairs-ou-medium-v1"].n_grid, defs["pairs-ou-medium-v1"].bar_interval_s) == (
            20_000,
            3_600,
        )
        assert (defs["pairs-ou-large-v1"].n_grid, defs["pairs-ou-large-v1"].bar_interval_s) == (
            200_000,
            60,
        )
        gappy = defs["pairs-ou-gappy-v1"]
        assert (gappy.n_grid, gappy.bar_interval_s, gappy.gap_fraction) == (5_000, 3_600, 0.02)
        control = defs["pairs-rw-control-v1"]
        assert isinstance(control.y_process, IndependentRandomWalk)
        assert (control.n_grid, control.bar_interval_s) == (2_000, 86_400)
        for name in ("pairs-ou-small-v1", "pairs-ou-medium-v1", "pairs-ou-large-v1", gappy.name):
            assert isinstance(defs[name].y_process, CointegratedOU)

    def test_golden_signatures_are_distinct_sha256(self) -> None:
        shas = [d.golden_sha256 for d in DATASET_DEFINITIONS.values()]
        assert len(set(shas)) == len(shas)
        assert all(len(s) == 64 and int(s, 16) >= 0 for s in shas)

    def test_registry_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            DATASET_DEFINITIONS["x"] = DATASET_DEFINITIONS[SMALL]  # type: ignore[index]

    def test_libm_constants_match(self) -> None:
        assert math.log(100.0) == ds.LN_100
        assert math.log(2.0) == ds.LN_2

    @pytest.mark.parametrize(
        "changes",
        [
            {"n_grid": 1},
            {"bar_interval_s": 0},
            {"gap_fraction": 1.0},
            {"seed": -1},
            {"start": datetime(2015, 1, 1)},  # naive datetime
            {"y_process": CointegratedOU(alpha_true=0, beta_true=1, half_life_s=10, sigma_s=1)},
        ],
    )
    def test_definition_validation(self, changes: dict[str, object]) -> None:
        with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
            dataclasses.replace(DATASET_DEFINITIONS[SMALL], **changes)  # type: ignore[arg-type]

    def test_dataset_ref_and_unknown_name(self) -> None:
        ref = dataset_ref(SMALL)
        assert ref == DatasetRef(name=SMALL, sha256=DATASET_DEFINITIONS[SMALL].golden_sha256)
        with pytest.raises(DataIntegrityError, match="unknown dataset"):
            dataset_ref("no-such-dataset")


class TestGenerator:
    @pytest.mark.parametrize("name", ALL_DATASETS)
    def test_regeneration_reproduces_golden_sha_bit_for_bit(self, name: str) -> None:
        defn = DATASET_DEFINITIONS[name]
        ts, cx, cy = generate_arrays(defn)
        assert dataset_content_sha256(ts, cx, cy) == defn.golden_sha256
        ts2, cx2, cy2 = generate_arrays(defn)
        assert np.array_equal(ts, ts2)
        assert np.array_equal(cx, cx2)
        assert np.array_equal(cy, cy2)

    @pytest.mark.parametrize("name", ALL_DATASETS)
    def test_generator_matches_documented_recursions(self, name: str) -> None:
        defn = DATASET_DEFINITIONS[name]
        ts, cx, cy = generate_arrays(defn)
        ets, ecx, ecy = _explicit_generate(defn)
        assert ts.tolist() == ets
        assert np.array_equal(cx, ecx)
        assert np.array_equal(cy, ecy)

    def test_first_bar_is_the_initial_state(self, generated: dict[str, Bars]) -> None:
        defn = DATASET_DEFINITIONS[SMALL]
        bars = generated[SMALL]
        assert isinstance(defn.y_process, CointegratedOU)
        assert bars.close_x[0] == 100.0
        expected_y = np.round(
            math.exp(defn.y_process.alpha_true + defn.y_process.beta_true * ds.LN_100), 6
        )
        assert bars.close_y[0] == expected_y
        assert int(bars.ts[0]) == int(defn.start.timestamp()) * 1_000_000_000

    def test_prices_are_quantised_to_the_tick(self, generated: dict[str, Bars]) -> None:
        for bars in generated.values():
            for col in (bars.close_x, bars.close_y):
                assert np.array_equal(col, np.round(col, 6))

    def test_gappy_dataset_has_deterministic_interior_gaps(
        self, generated: dict[str, Bars]
    ) -> None:
        defn = DATASET_DEFINITIONS["pairs-ou-gappy-v1"]
        bars = generated[defn.name]
        step = defn.bar_interval_s * 1_000_000_000
        missing = defn.n_grid - len(bars)
        assert 0.01 * defn.n_grid < missing < 0.03 * defn.n_grid
        diffs = np.diff(bars.ts)
        assert np.all(diffs % step == 0)
        assert int(diffs.max()) > step
        grid_end = defn.start_ns + (defn.n_grid - 1) * step
        assert int(bars.ts[0]) == defn.start_ns
        assert int(bars.ts[-1]) == grid_end
        for name in ALL_DATASETS:
            if name != defn.name:
                assert len(generated[name]) == DATASET_DEFINITIONS[name].n_grid

    @pytest.mark.parametrize(
        "name",
        ["pairs-ou-small-v1", "pairs-ou-medium-v1", "pairs-ou-large-v1", "pairs-ou-gappy-v1"],
    )
    def test_ou_spread_has_the_documented_mean_reversion(
        self, name: str, generated: dict[str, Bars]
    ) -> None:
        defn = DATASET_DEFINITIONS[name]
        proc = defn.y_process
        assert isinstance(proc, CointegratedOU)
        bars = generated[name]
        spread = np.log(bars.close_y) - proc.alpha_true - proc.beta_true * np.log(bars.close_x)
        theta_dt = ds.LN_2 / proc.half_life_s * defn.bar_interval_s
        assert _ar1(spread) == pytest.approx(1.0 - theta_dt, abs=0.02)

    def test_control_pair_is_not_cointegrated(self, generated: dict[str, Bars]) -> None:
        bars = generated["pairs-rw-control-v1"]
        lx, ly = np.log(bars.close_x), np.log(bars.close_y)
        beta = float(np.cov(lx, ly, ddof=0)[0, 1] / np.var(lx))
        resid = ly - beta * lx
        assert _ar1(resid) > 0.99  # unit-root-like residual: a negative control

    def test_generate_bars_is_read_only_and_verified(self, generated: dict[str, Bars]) -> None:
        bars = generated[SMALL]
        assert bars.sha256 == DATASET_DEFINITIONS[SMALL].golden_sha256
        for arr in (bars.ts, bars.close_x, bars.close_y):
            assert not arr.flags.writeable

    def test_generate_bars_detects_drift(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = dataclasses.replace(DATASET_DEFINITIONS[SMALL], golden_sha256="0" * 64)
        monkeypatch.setattr(ds, "get_definition", lambda name: bad)
        with pytest.raises(DataIntegrityError, match="does not match its recorded signature"):
            generate_bars(SMALL)


# --------------------------------------------------------------------------------------
# Content hash and validation
# --------------------------------------------------------------------------------------


class TestContentHash:
    def test_hash_follows_documented_byte_layout(self, generated: dict[str, Bars]) -> None:
        b = generated[SMALL]
        header = canonical_json(
            {
                "columns": ["ts", "close_x", "close_y"],
                "dtypes": ["<i8", "<f8", "<f8"],
                "length": len(b),
            }
        )
        payload = (
            header.encode()
            + b"\0"
            + b.ts.astype("<i8").tobytes()
            + b.close_x.astype("<f8").tobytes()
            + b.close_y.astype("<f8").tobytes()
        )
        assert (
            dataset_content_sha256(b.ts, b.close_x, b.close_y)
            == hashlib.sha256(payload).hexdigest()
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"compression": "none"},
            {"compression": "snappy", "row_group_size": 97},
            {"compression": "zstd", "use_dictionary": False, "data_page_size": 1024},
            {"compression": "gzip", "data_page_version": "2.0"},
        ],
    )
    def test_hash_is_independent_of_parquet_encoding(
        self, tmp_path: Path, generated: dict[str, Bars], kwargs: dict[str, object]
    ) -> None:
        b = generated[SMALL]
        path = tmp_path / "encoded.parquet"
        _write_table(path, b.ts, b.close_x, b.close_y, **kwargs)
        ts, cx, cy, _meta = ds._read_parquet(path)
        assert dataset_content_sha256(ts, cx, cy) == b.sha256

    def test_hash_detects_a_single_ulp_change(self, generated: dict[str, Bars]) -> None:
        b = generated[SMALL]
        cy = b.close_y.copy()
        cy[1234] = np.nextafter(cy[1234], np.inf)
        assert dataset_content_sha256(b.ts, b.close_x, cy) != b.sha256

    def test_hash_depends_on_column_order(self, generated: dict[str, Bars]) -> None:
        b = generated[SMALL]
        assert dataset_content_sha256(b.ts, b.close_y, b.close_x) != b.sha256

    def test_hash_rejects_wrong_dtypes(self, generated: dict[str, Bars]) -> None:
        b = generated[SMALL]
        with pytest.raises(DataIntegrityError, match="dtype"):
            dataset_content_sha256(b.ts.astype(np.float64), b.close_x, b.close_y)  # type: ignore[arg-type]
        with pytest.raises(DataIntegrityError, match="dtype"):
            dataset_content_sha256(b.ts, b.close_x.astype(np.float32), b.close_y)  # type: ignore[arg-type]


def _good_bars() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.arange(5, dtype=np.int64) * 60_000_000_000
    return ts, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([5.0, 4.0, 3.0, 2.0, 1.0])


class TestValidateBars:
    def test_valid_bars_pass(self) -> None:
        validate_bars(*_good_bars())

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda ts, x, y: (ts[:0], x[:0], y[:0]), "empty"),
            (lambda ts, x, y: (ts, x[:-1], y), "length mismatch"),
            (lambda ts, x, y: (ts.astype(np.float64), x, y), "dtype"),
            (lambda ts, x, y: (ts.astype(np.int32), x, y), "dtype"),
            (lambda ts, x, y: (ts, x.astype(np.float32), y), "dtype"),
            (lambda ts, x, y: (ts.reshape(5, 1), x, y), "1-D"),
            (lambda ts, x, y: (ts[::-1].copy(), x, y), "decreasing"),
            (lambda ts, x, y: (np.array([0, 1, 1, 2, 3], dtype=np.int64), x, y), "duplicate"),
            (lambda ts, x, y: (ts, np.array([1.0, np.nan, 3, 4, 5]), y), "non-finite"),
            (lambda ts, x, y: (ts, x, np.array([1.0, 2, np.inf, 4, 5])), "non-finite"),
            (lambda ts, x, y: (ts, x, np.array([1.0, 2, -np.inf, 4, 5])), "non-finite"),
            (lambda ts, x, y: (ts, np.array([1.0, 2, 3, 0.0, 5]), y), "non-positive"),
            (lambda ts, x, y: (ts, x, np.array([1.0, -2, 3, 4, 5])), "non-positive"),
        ],
    )
    def test_invalid_bars_raise(self, mutate, match: str) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(DataIntegrityError, match=match):
            validate_bars(*mutate(*_good_bars()))

    def test_error_is_non_retryable(self) -> None:
        with pytest.raises(DataIntegrityError) as info:
            validate_bars(np.array([], dtype=np.int64), np.array([]), np.array([]))
        assert info.value.retryable is False


# --------------------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------------------


class TestCatalog:
    def test_catalog_entries_describe_the_generated_data(self, generated: dict[str, Bars]) -> None:
        catalog = get_catalog()
        assert [info.name for info in catalog.list()] == ALL_DATASETS
        for name, bars in generated.items():
            info = catalog.get(name)
            assert info is not None
            assert info.sha256 == bars.sha256
            assert info.n_bars == len(bars)
            assert info.start == ds.ns_to_datetime(int(bars.ts[0]))
            assert info.end == ds.ns_to_datetime(int(bars.ts[-1]))
            assert info.start.tzinfo is not None
            assert info.columns == BAR_COLUMNS
            assert info.synthetic is True
            assert info.license == DATASET_LICENSE
            assert info.bar_interval_s == DATASET_DEFINITIONS[name].bar_interval_s
            assert info.generator["seed"] == DATASET_DEFINITIONS[name].seed
            assert info.generator["kind"] in {"cointegrated_ou", "independent_rw"}
            canonical_json(info.model_dump(mode="json"))  # JSON-serialisable

    def test_unknown_name_and_caching(self) -> None:
        assert get_catalog().get("nope") is None
        assert get_catalog() is get_catalog()
        assert isinstance(get_catalog(), Catalog)

    def test_policy_service_uses_the_catalog(self) -> None:
        service = PolicyService(get_catalog())
        spec = ExperimentBatchSpec(
            name="catalog check",
            dataset=dataset_ref(SMALL),
            strategy=StrategyRef(name="pairs_zscore", version="1"),
            grid={"lookback": [20, 40]},
            fixed_params={"entry_z": 2.0, "exit_z": 0.5},
        )
        assert service.evaluate(spec, QuotaPolicy()).allowed
        tampered = spec.model_copy(update={"dataset": DatasetRef(name=SMALL, sha256="f" * 64)})
        decision = service.evaluate(tampered, QuotaPolicy())
        assert not decision.allowed
        assert {v.code for v in decision.violations} == {"DATASET_SIGNATURE_MISMATCH"}


# --------------------------------------------------------------------------------------
# Materialisation and loading
# --------------------------------------------------------------------------------------


class TestMaterializeAndLoad:
    def test_materialize_writes_a_verified_parquet_file(self, tmp_path: Path) -> None:
        path = materialize(SMALL, tmp_path)
        assert path == dataset_path(SMALL, tmp_path) == tmp_path / "datasets" / f"{SMALL}.parquet"
        assert os.listdir(path.parent) == [path.name]  # no temp files left behind
        table = pq.read_table(path)
        assert table.schema.field("ts").type == pa.timestamp("ns", tz="UTC")
        assert table.schema.field("close_x").type == pa.float64()
        meta = table.schema.metadata
        defn = DATASET_DEFINITIONS[SMALL]
        assert meta[ds.META_SHA256].decode() == defn.golden_sha256
        assert meta[ds.META_GENERATOR].decode() == canonical_json(defn.generator_params())
        assert meta[ds.META_NAME].decode() == SMALL

    def test_materialize_is_idempotent(self, tmp_path: Path) -> None:
        path = materialize(SMALL, tmp_path)
        before = path.stat()
        time.sleep(0.01)
        assert materialize(SMALL, tmp_path) == path
        after = path.stat()
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)

    def test_materialize_repairs_an_invalid_file(self, tmp_path: Path) -> None:
        path = dataset_path(SMALL, tmp_path)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a parquet file")
        materialize(SMALL, tmp_path)
        bars = load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=False)
        assert bars.sha256 == DATASET_DEFINITIONS[SMALL].golden_sha256

    def test_materialize_refuses_to_write_drifted_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = dataclasses.replace(DATASET_DEFINITIONS[SMALL], golden_sha256="0" * 64)
        monkeypatch.setattr(ds, "get_definition", lambda name: bad)
        with pytest.raises(DataIntegrityError, match="refusing to materialise"):
            materialize(SMALL, tmp_path)
        assert not (tmp_path / "datasets").exists()  # nothing written, not even a temp file

    def test_concurrent_materialize_is_safe(self, tmp_path: Path) -> None:
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                materialize("pairs-ou-gappy-v1", tmp_path)
            except BaseException as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert os.listdir(tmp_path / "datasets") == ["pairs-ou-gappy-v1.parquet"]
        load_bars(dataset_ref("pairs-ou-gappy-v1"), tmp_path, auto_materialize=False)

    def test_load_bars_round_trip(self, tmp_path: Path, generated: dict[str, Bars]) -> None:
        materialize(SMALL, tmp_path)
        bars = load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=False)
        ref = generated[SMALL]
        assert bars.name == SMALL
        assert bars.sha256 == ref.sha256
        assert np.array_equal(bars.ts, ref.ts)
        assert np.array_equal(bars.close_x, ref.close_x)
        assert np.array_equal(bars.close_y, ref.close_y)
        assert bars.ts.dtype == np.int64
        assert bars.close_x.dtype == np.float64
        assert not bars.close_y.flags.writeable

    def test_auto_materialize_controls_missing_files(self, tmp_path: Path) -> None:
        ref = dataset_ref("pairs-rw-control-v1")
        with pytest.raises(DataIntegrityError, match="missing"):
            load_bars(ref, tmp_path, auto_materialize=False)
        bars = load_bars(ref, tmp_path, auto_materialize=True)
        assert bars.sha256 == ref.sha256
        assert dataset_path(ref.name, tmp_path).exists()

    def test_unknown_dataset(self, tmp_path: Path) -> None:
        with pytest.raises(DataIntegrityError, match="unknown dataset"):
            load_bars(
                DatasetRef(name="mystery-v1", sha256="a" * 64), tmp_path, auto_materialize=True
            )

    def test_requested_signature_must_match_registry(self, tmp_path: Path) -> None:
        ref = DatasetRef(name=SMALL, sha256=DATASET_DEFINITIONS["pairs-ou-medium-v1"].golden_sha256)
        with pytest.raises(DataIntegrityError, match="requested signature"):
            load_bars(ref, tmp_path, auto_materialize=True)
        assert not dataset_path(SMALL, tmp_path).exists()

    def test_corrupt_file_is_rejected_even_with_auto_materialize(self, tmp_path: Path) -> None:
        path = materialize(SMALL, tmp_path)
        data = bytearray(path.read_bytes())
        data[len(data) // 2 : len(data) // 2 + 64] = b"\xff" * 64
        path.write_bytes(bytes(data))
        with pytest.raises(DataIntegrityError):
            load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=True)

    def test_truncated_file_is_rejected(self, tmp_path: Path) -> None:
        path = materialize(SMALL, tmp_path)
        path.write_bytes(path.read_bytes()[:100])
        with pytest.raises(DataIntegrityError, match="cannot read"):
            load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=False)

    def test_tampered_content_fails_the_checksum(
        self, tmp_path: Path, generated: dict[str, Bars]
    ) -> None:
        b = generated[SMALL]
        cx = b.close_x.copy()
        cx[100] += 0.01  # still a valid, positive price
        path = dataset_path(SMALL, tmp_path)
        _write_table(path, b.ts, cx, b.close_y)
        with pytest.raises(DataIntegrityError, match="checksum mismatch"):
            load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=True)
        # verify=False is the documented trust mode: structural checks only.
        trusted = load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=False, verify=False)
        assert trusted.close_x[100] == cx[100]

    def test_declared_signature_must_match(
        self, tmp_path: Path, generated: dict[str, Bars]
    ) -> None:
        b = generated[SMALL]
        path = dataset_path(SMALL, tmp_path)
        table = pa.table(
            {
                "ts": pa.array(b.ts, type=pa.timestamp("ns", tz="UTC")),
                "close_x": pa.array(b.close_x),
                "close_y": pa.array(b.close_y),
            }
        ).replace_schema_metadata({ds.META_SHA256: b"e" * 64})
        path.parent.mkdir(parents=True)
        pq.write_table(table, path)
        with pytest.raises(DataIntegrityError, match="declares signature"):
            load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=False, verify=False)

    @pytest.mark.parametrize("variant", ["int64-ts", "microsecond-ts", "extra-column"])
    def test_wrong_schema_is_rejected(
        self, tmp_path: Path, generated: dict[str, Bars], variant: str
    ) -> None:
        b = generated[SMALL]
        path = dataset_path(SMALL, tmp_path)
        path.parent.mkdir(parents=True)
        columns: dict[str, object] = {"ts": b.ts, "close_x": b.close_x, "close_y": b.close_y}
        if variant == "microsecond-ts":
            # Parquet format 2.4 cannot store ns timestamps: pyarrow coerces them to us.
            columns["ts"] = pa.array(b.ts, type=pa.timestamp("ns", tz="UTC"))
            pq.write_table(pa.table(columns), path, version="2.4")
        else:
            if variant == "extra-column":
                columns["ts"] = pa.array(b.ts, type=pa.timestamp("ns", tz="UTC"))
                columns["volume"] = b.close_x
            pq.write_table(pa.table(columns), path)
        with pytest.raises(DataIntegrityError, match="unexpected schema"):
            load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=False)

    def test_nulls_are_rejected(self, tmp_path: Path, generated: dict[str, Bars]) -> None:
        b = generated[SMALL]
        path = dataset_path(SMALL, tmp_path)
        path.parent.mkdir(parents=True)
        cx = pa.array([*b.close_x.tolist()[:-1], None], type=pa.float64())
        table = pa.table(
            {
                "ts": pa.array(b.ts, type=pa.timestamp("ns", tz="UTC")),
                "close_x": cx,
                "close_y": b.close_y,
            }
        )
        pq.write_table(table, path)
        with pytest.raises(DataIntegrityError, match="nulls"):
            load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=False)

    def test_invalid_rows_are_rejected(self, tmp_path: Path, generated: dict[str, Bars]) -> None:
        b = generated[SMALL]
        ts = b.ts.copy()
        ts[10] = ts[9]
        _write_table(dataset_path(SMALL, tmp_path), ts, b.close_x, b.close_y)
        with pytest.raises(DataIntegrityError, match="duplicate"):
            load_bars(dataset_ref(SMALL), tmp_path, auto_materialize=False)


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------


def _fake_bars(tag: str) -> Bars:
    ts, x, y = _good_bars()
    return Bars(
        name=tag, ts=ts, close_x=x, close_y=y, sha256=hashlib.sha256(tag.encode()).hexdigest()
    )


class TestBarsCache:
    def test_lru_eviction_order(self) -> None:
        cache = BarsCache(capacity=2)
        a, b, c = _fake_bars("a"), _fake_bars("b"), _fake_bars("c")
        cache.put(a)
        cache.put(b)
        assert cache.get(a.sha256) is a  # refresh a -> b is now least recent
        cache.put(c)
        assert cache.get(b.sha256) is None
        assert cache.get(a.sha256) is a
        assert cache.get(c.sha256) is c
        assert len(cache) == 2
        assert cache.stats()["evictions"] == 1
        cache.clear()
        assert len(cache) == 0

    def test_capacity_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            BarsCache(capacity=0)

    def test_get_or_load_hits_after_first_load(self, tmp_path: Path) -> None:
        cache = BarsCache()
        ref = dataset_ref(SMALL)
        first = cache.get_or_load(ref, tmp_path, auto_materialize=True)
        second = cache.get_or_load(ref, tmp_path, auto_materialize=False)
        assert first is second
        stats = cache.stats()
        assert (stats["hits"], stats["misses"], stats["size"]) == (1, 1, 1)

    def test_mismatched_name_is_not_served_from_cache(self, tmp_path: Path) -> None:
        cache = BarsCache()
        small = dataset_ref(SMALL)
        cache.get_or_load(small, tmp_path, auto_materialize=True)
        bogus = DatasetRef(name="pairs-ou-medium-v1", sha256=small.sha256)
        with pytest.raises(DataIntegrityError, match="requested signature"):
            cache.get_or_load(bogus, tmp_path, auto_materialize=True)

    def test_concurrent_loads_are_single_flight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        materialize(SMALL, tmp_path)
        calls: list[int] = []
        real_load = ds.load_bars

        def slow_load(*args: object, **kwargs: object) -> Bars:
            calls.append(1)
            time.sleep(0.05)
            return real_load(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(ds, "load_bars", slow_load)
        cache = BarsCache()
        results: list[Bars] = []
        lock = threading.Lock()

        def worker() -> None:
            bars = cache.get_or_load(dataset_ref(SMALL), tmp_path, auto_materialize=False)
            with lock:
                results.append(bars)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1
        assert len(results) == 8
        assert all(r is results[0] for r in results)
        assert cache.stats()["hits"] == 7
