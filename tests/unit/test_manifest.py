"""Unit tests for the reproducibility manifest."""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest

import quantgrid
from quantgrid import manifest as mf
from quantgrid.domain.hashing import is_sha256_hex
from quantgrid.domain.specs import RESULT_SCHEMA_VERSION, DatasetInfo

SHA = "c" * 64


@pytest.fixture
def ds() -> DatasetInfo:
    return DatasetInfo(
        name="ds", sha256=SHA, n_bars=500, start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2020, 1, 2, tzinfo=UTC), bar_interval_s=60, columns=("a", "b"),
        synthetic=True, license="synthetic", description="d", generator={"seed": 1},
    )  # fmt: skip


def test_code_digest_stable_hex() -> None:
    d = mf.code_digest()
    assert is_sha256_hex(d)
    assert mf.code_digest() == d


def test_code_digest_matches_independent_computation() -> None:
    import hashlib

    h = hashlib.sha256()
    for path in sorted(mf.PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(mf.PACKAGE_ROOT).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    mf.code_digest.cache_clear()
    try:
        assert mf.code_digest() == h.hexdigest()
    finally:
        mf.code_digest.cache_clear()


def test_code_digest_changes_when_package_changes(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "sub" / "b.py").write_text("y = 2\n")
    monkeypatch.setattr(mf, "PACKAGE_ROOT", pkg)
    mf.code_digest.cache_clear()
    try:
        d1 = mf.code_digest()
        mf.code_digest.cache_clear()
        assert mf.code_digest() == d1
        (pkg / "sub" / "b.py").write_text("y = 3\n")
        mf.code_digest.cache_clear()
        d2 = mf.code_digest()
        assert d2 != d1
        (pkg / "__pycache__").mkdir()
        (pkg / "__pycache__" / "junk.py").write_text("zzz")
        (pkg / "notes.txt").write_text("ignored")
        mf.code_digest.cache_clear()
        assert mf.code_digest() == d2
        (pkg / "a.py").rename(pkg / "c.py")  # rename changes the digest (path is hashed)
        mf.code_digest.cache_clear()
        assert mf.code_digest() != d2
    finally:
        mf.code_digest.cache_clear()


def test_environment_manifest_contents() -> None:
    env = mf.environment_manifest()
    assert re.match(r"^3\.\d+\.\d+", env["python"])
    assert env["machine"]
    assert env["implementation"]
    assert isinstance(env["packages"], dict) and env["packages"]
    assert "pydantic" in env["packages"] and "numpy" in env["packages"]
    assert list(env["packages"]) == sorted(env["packages"])
    assert is_sha256_hex(env["env_sha"])
    assert mf.env_sha() == env["env_sha"]
    assert "lockfile_sha" in env
    assert env["lockfile_sha"] is None or is_sha256_hex(env["lockfile_sha"])
    json.dumps(env)


def test_env_sha_excludes_platform_string_but_covers_packages() -> None:
    from quantgrid.domain.hashing import canonical_sha256

    env = mf.environment_manifest()
    hashed = {k: v for k, v in env.items() if k not in ("platform", "env_sha", "lockfile_sha")}
    assert canonical_sha256(hashed) == env["env_sha"]
    hashed["packages"] = {**hashed["packages"], "zzz-extra": "1"}
    assert canonical_sha256(hashed) != env["env_sha"]


def test_lockfile_sha_not_part_of_env_sha() -> None:
    env = mf.environment_manifest()
    # lockfile_sha is added after env_sha is computed, so it cannot influence it.
    assert "lockfile_sha" not in {k for k in env if k == "env_sha"}


def test_build_manifest_shape(ds: DatasetInfo) -> None:
    m = mf.build_manifest(ds, "d" * 64)
    assert m["manifest_version"] == mf.MANIFEST_VERSION
    assert m["spec_sha"] == "d" * 64
    assert m["result_schema_version"] == RESULT_SCHEMA_VERSION
    assert m["code"]["code_sha"] == mf.code_digest()
    assert m["code"]["package_version"] == quantgrid.__version__
    assert {"git_commit", "git_dirty", "source"} <= set(m["code"])
    assert m["environment"]["env_sha"] == mf.env_sha()
    assert m["dataset"]["sha256"] == SHA and m["dataset"]["name"] == "ds"
    assert m["dataset"]["start"] == "2020-01-01T00:00:00Z"
    assert m["dataset"]["columns"] == ["a", "b"]
    assert m["created_at"].endswith("Z") and "+" not in m["created_at"]
    datetime.fromisoformat(m["created_at"].replace("Z", "+00:00"))
    json.dumps(m)  # must be JSON-serialisable (stored in a JSONB column)


def test_git_commit_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QG_GIT_COMMIT", "deadbeef")
    mf.git_info.cache_clear()
    try:
        assert mf.git_info() == {"git_commit": "deadbeef", "git_dirty": None, "source": "env"}
    finally:
        mf.git_info.cache_clear()


def test_manifest_drift_no_drift(ds: DatasetInfo) -> None:
    drift = mf.manifest_drift(mf.build_manifest(ds, "d" * 64))
    assert drift["code_sha_match"] is True
    assert drift["env_sha_match"] is True
    assert drift["changed_packages"] == {}
    assert drift["recorded_code_sha"] == drift["current_code_sha"] == mf.code_digest()


def test_manifest_drift_detects_changes(ds: DatasetInfo) -> None:
    rec = copy.deepcopy(mf.build_manifest(ds, "d" * 64))
    some_pkg = next(iter(rec["environment"]["packages"]))
    old_version = rec["environment"]["packages"][some_pkg]
    rec["environment"]["packages"][some_pkg] = "0.0.0-doctored"
    rec["environment"]["packages"]["ghost-package"] = "9.9"
    del rec["environment"]["packages"]["numpy"]
    rec["environment"]["env_sha"] = "0" * 64
    rec["environment"]["python"] = "2.7.18"
    rec["code"]["code_sha"] = "f" * 64
    drift = mf.manifest_drift(rec)
    assert drift["code_sha_match"] is False
    assert drift["recorded_code_sha"] == "f" * 64
    assert drift["current_code_sha"] == mf.code_digest()
    assert drift["env_sha_match"] is False
    assert drift["python"] == {"recorded": "2.7.18", "current": mf.environment_manifest()["python"]}
    ch = drift["changed_packages"]
    assert ch[some_pkg] == {"recorded": "0.0.0-doctored", "current": old_version}
    assert ch["ghost-package"] == {"recorded": "9.9", "current": None}
    assert ch["numpy"]["recorded"] is None and ch["numpy"]["current"]
    assert set(ch) == {some_pkg, "ghost-package", "numpy"}


def test_manifest_drift_tolerates_empty_record() -> None:
    drift = mf.manifest_drift({})
    assert drift["code_sha_match"] is False
    assert drift["env_sha_match"] is False
    assert drift["recorded_code_sha"] is None
    # every current package shows as changed relative to an empty record
    assert set(drift["changed_packages"]) == set(mf.installed_packages())
