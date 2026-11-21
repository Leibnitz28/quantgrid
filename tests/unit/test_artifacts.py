"""Tests for the content-addressed artifact store (``quantgrid.storage.artifacts``)."""

from __future__ import annotations

import errno
import hashlib
import logging
import multiprocessing as mp
import os
import signal
import stat
import threading
import time
from collections.abc import Callable
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Barrier as MPBarrier
from pathlib import Path
from typing import Any

import pytest

from quantgrid.config import Settings
from quantgrid.domain.errors import ArtifactError
from quantgrid.storage.artifacts import (
    TEMP_SUFFIX,
    ArtifactIntegrityError,
    ArtifactStore,
    LocalArtifactStore,
    PublishedArtifact,
    StagedArtifact,
    artifact_uri_for,
    sha256_from_uri,
)

PAYLOAD = b"PAR1" + bytes(range(256)) * 64 + b"PAR1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _temp_files(store: LocalArtifactStore) -> list[Path]:
    tmp = store.root / "tmp"
    return sorted(tmp.iterdir()) if tmp.exists() else []


def _object_files(store: LocalArtifactStore) -> list[Path]:
    objects = store.root / "sha256"
    return sorted(p for p in objects.rglob("*") if p.is_file()) if objects.exists() else []


def _overwrite(path: Path, data: bytes) -> None:
    """Simulate on-disk corruption of a (read-only) file."""
    path.chmod(0o644)
    path.write_bytes(data)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts")


def test_stage_writes_complete_temp_file(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="task-1-attempt-2")

    assert staged.sha256 == _sha(PAYLOAD)
    assert staged.size == len(PAYLOAD)
    assert staged.temp_path.parent == store.root / "tmp"
    assert staged.temp_path.name.startswith("task-1-attempt-2.")
    assert staged.temp_path.name.endswith(TEMP_SUFFIX)
    assert staged.temp_uri == f"artifact://tmp/{staged.temp_path.name}"
    assert staged.temp_path.read_bytes() == PAYLOAD
    assert not store.exists(store.uri_for(staged.sha256))  # not published yet


def test_stage_uses_unique_temp_names(store: LocalArtifactStore) -> None:
    a = store.stage(PAYLOAD, hint="same")
    b = store.stage(PAYLOAD, hint="same")
    assert a.temp_path != b.temp_path
    assert len(_temp_files(store)) == 2


@pytest.mark.parametrize(
    ("hint", "expected_prefix"),
    [
        ("../../etc/passwd", "_.._etc_passwd."),  # one path component, no separators
        ("/abs/path", "_abs_path."),
        ("..", "artifact."),
        ("", "artifact."),
        ("spaces and\nnewline", "spaces_and_newline."),
        ("-leading-dash", "leading-dash."),
        ("x" * 500, "x" * 64 + "."),
    ],
)
def test_stage_sanitises_hint(store: LocalArtifactStore, hint: str, expected_prefix: str) -> None:
    staged = store.stage(b"data", hint=hint)
    assert staged.temp_path.parent == store.root / "tmp"
    assert staged.temp_path.name.startswith(expected_prefix)
    assert "/" not in staged.temp_path.name
    assert not staged.temp_path.name.startswith((".", "-"))
    assert staged.temp_path.exists()


def test_stage_rejects_non_bytes(store: LocalArtifactStore) -> None:
    with pytest.raises(TypeError):
        store.stage("text", hint="x")  # type: ignore[arg-type]


def test_commit_publishes_content_addressed_read_only_file(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="t")
    published = store.commit(staged)

    sha = _sha(PAYLOAD)
    assert published == PublishedArtifact(
        uri=f"artifact://sha256/{sha[:2]}/{sha}.parquet",
        sha256=sha,
        size=len(PAYLOAD),
        created=True,
    )
    final = store.root / "sha256" / sha[:2] / f"{sha}.parquet"
    assert store.path_for_uri(published.uri) == final
    assert final.read_bytes() == PAYLOAD
    assert stat.S_IMODE(final.stat().st_mode) & 0o222 == 0  # immutable: no write bits
    assert not staged.temp_path.exists()
    assert _temp_files(store) == []
    assert _object_files(store) == [final]


def test_empty_payload_round_trips(store: LocalArtifactStore) -> None:
    published = store.commit(store.stage(b"", hint="empty"))
    assert published.sha256 == _sha(b"")
    assert published.size == 0
    assert store.read_bytes(published.uri) == b""


def test_duplicate_commit_is_idempotent(store: LocalArtifactStore) -> None:
    first = store.commit(store.stage(PAYLOAD, hint="attempt-1"))
    second_staged = store.stage(PAYLOAD, hint="attempt-2")
    final = store.path_for_uri(first.uri)
    inode_before = final.stat().st_ino

    second = store.commit(second_staged)

    assert first.created is True
    assert second.created is False
    assert second.repaired is False
    assert (second.uri, second.sha256, second.size) == (first.uri, first.sha256, first.size)
    assert final.stat().st_ino == inode_before  # the published file was not touched
    assert not second_staged.temp_path.exists()
    assert _temp_files(store) == []
    assert _object_files(store) == [final]


def test_committing_the_same_staged_artifact_twice_is_idempotent(
    store: LocalArtifactStore,
) -> None:
    staged = store.stage(PAYLOAD, hint="t")
    assert store.commit(staged).created is True
    again = store.commit(staged)  # e.g. a retry after a timeout around the first commit
    assert again.created is False
    assert store.verify(again.uri, _sha(PAYLOAD))


def test_different_content_never_overwrites_existing_artifact(store: LocalArtifactStore) -> None:
    a = store.commit(store.stage(b"result A", hint="t"))
    b = store.commit(store.stage(b"result B", hint="t"))
    assert a.uri != b.uri
    assert store.read_bytes(a.uri) == b"result A"
    assert store.read_bytes(b.uri) == b"result B"


def test_discard_removes_temp_and_is_idempotent(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="t")
    store.discard(staged)
    store.discard(staged)
    assert not staged.temp_path.exists()
    assert not store.exists(store.uri_for(staged.sha256))


def test_discard_after_commit_keeps_published_artifact(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="t")
    published = store.commit(staged)
    store.discard(staged)
    assert store.verify(published.uri, published.sha256)


def test_commit_after_discard_fails_cleanly(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="t")
    store.discard(staged)
    with pytest.raises(ArtifactError, match="no longer exists"):
        store.commit(staged)
    assert _object_files(store) == []


def test_commit_rejects_temp_file_modified_after_staging(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="t")
    _overwrite(staged.temp_path, PAYLOAD[:-1] + b"X")

    with pytest.raises(ArtifactIntegrityError, match="changed after staging") as excinfo:
        store.commit(staged)

    assert isinstance(excinfo.value, ArtifactError)
    assert excinfo.value.retryable is True
    assert not staged.temp_path.exists()  # the bad temp file is dropped
    assert _object_files(store) == []  # nothing was published


def test_commit_rejects_truncated_temp_file(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="t")
    _overwrite(staged.temp_path, PAYLOAD[:100])
    with pytest.raises(ArtifactIntegrityError):
        store.commit(staged)
    assert _object_files(store) == []


def test_commit_and_discard_reject_foreign_staged_artifacts(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    outside = tmp_path / "victim.partial"
    outside.write_bytes(PAYLOAD)
    foreign = StagedArtifact(
        temp_path=outside, temp_uri="artifact://tmp/victim.partial", sha256=_sha(PAYLOAD), size=1
    )
    with pytest.raises(ValueError, match="does not belong"):
        store.commit(foreign)
    with pytest.raises(ValueError, match="does not belong"):
        store.discard(foreign)
    assert outside.exists()

    other_store = LocalArtifactStore(tmp_path / "other")
    staged_elsewhere = other_store.stage(PAYLOAD, hint="t")
    with pytest.raises(ValueError, match="does not belong"):
        store.commit(staged_elsewhere)


def test_commit_rejects_malformed_staged_digest(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="t")
    bad = StagedArtifact(staged.temp_path, staged.temp_uri, sha256="ABC", size=staged.size)
    with pytest.raises(ValueError, match="malformed"):
        store.commit(bad)


def test_crash_between_stage_and_commit_leaves_only_temp_file(store: LocalArtifactStore) -> None:
    staged = store.stage(PAYLOAD, hint="crashed-attempt")
    # ... the worker dies here: commit never runs ...
    uri = store.uri_for(staged.sha256)
    assert staged.temp_path.read_bytes() == PAYLOAD
    assert not store.exists(uri)
    assert list(store.iter_artifacts()) == []

    assert store.gc_temp(older_than_s=3600) == 0  # too young: could be a live worker
    assert staged.temp_path.exists()

    old = time.time() - 7200
    os.utime(staged.temp_path, (old, old))
    assert store.gc_temp(older_than_s=3600) == 1
    assert _temp_files(store) == []
    assert not store.exists(uri)


def test_gc_temp_only_removes_partial_files(store: LocalArtifactStore) -> None:
    published = store.commit(store.stage(PAYLOAD, hint="kept"))
    stale = store.stage(b"stale", hint="stale")
    unrelated = store.root / "tmp" / "README.txt"
    unrelated.write_text("not a temp artifact")
    old = time.time() - 600
    for path in (stale.temp_path, unrelated, store.path_for_uri(published.uri)):
        os.utime(path, (old, old))

    assert store.gc_temp(older_than_s=60) == 1
    assert not stale.temp_path.exists()
    assert unrelated.exists()
    assert store.verify(published.uri, published.sha256)


def test_gc_temp_without_tmp_dir_and_argument_validation(store: LocalArtifactStore) -> None:
    assert store.gc_temp(0) == 0
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="older_than_s"):
            store.gc_temp(bad)


def _stage_then_die(root: str, payload: bytes) -> None:
    """Child process: stage an artifact, then die abruptly before committing."""
    LocalArtifactStore(Path(root)).stage(payload, hint="killed-worker")
    os.kill(os.getpid(), signal.SIGKILL)


def test_process_killed_between_stage_and_commit(store: LocalArtifactStore) -> None:
    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_stage_then_die, args=(str(store.root), PAYLOAD))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == -signal.SIGKILL

    temps = _temp_files(store)
    assert len(temps) == 1
    assert temps[0].name.startswith("killed-worker.")
    assert temps[0].read_bytes() == PAYLOAD  # fsynced before stage() returned
    assert not store.exists(store.uri_for(_sha(PAYLOAD)))  # never a partial artifact

    old = time.time() - 3600
    os.utime(temps[0], (old, old))
    assert store.gc_temp(older_than_s=60) == 1
    assert _temp_files(store) == []

    # A replacement attempt publishes normally afterwards.
    published = store.commit(store.stage(PAYLOAD, hint="replacement"))
    assert published.created is True


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda p: _overwrite(p, b"garbage"), id="overwritten"),
        pytest.param(lambda p: _overwrite(p, p.read_bytes()[:10]), id="truncated"),
        pytest.param(
            lambda p: _overwrite(p, bytes([p.read_bytes()[0] ^ 0xFF]) + p.read_bytes()[1:]),
            id="bit-flip-same-size",
        ),
    ],
)
def test_corrupted_final_file_is_repaired(
    store: LocalArtifactStore,
    caplog: pytest.LogCaptureFixture,
    corrupt: Callable[[Path], None],
) -> None:
    published = store.commit(store.stage(PAYLOAD, hint="first"))
    final = store.path_for_uri(published.uri)
    corrupt(final)
    assert not store.verify(published.uri, published.sha256)

    with caplog.at_level(logging.WARNING, logger="quantgrid.storage.artifacts"):
        repaired = store.commit(store.stage(PAYLOAD, hint="second"))

    assert repaired.created is True
    assert repaired.repaired is True
    assert repaired.uri == published.uri
    assert final.read_bytes() == PAYLOAD
    assert store.verify(published.uri, published.sha256, expected_size=len(PAYLOAD))
    assert _temp_files(store) == []
    assert any("corrupt artifact" in r.getMessage() for r in caplog.records)


def test_symlink_at_artifact_path_is_never_followed_and_is_replaced(
    store: LocalArtifactStore, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_bytes(PAYLOAD)  # same bytes, so a naive hash check would pass
    uri = store.uri_for(_sha(PAYLOAD))
    final = store.path_for_uri(uri)
    final.parent.mkdir(parents=True)
    final.symlink_to(secret)

    assert not store.exists(uri)
    assert not store.verify(uri, _sha(PAYLOAD))
    with pytest.raises(OSError):  # noqa: PT011 - the errno depends on the platform
        store.read_bytes(uri)

    published = store.commit(store.stage(PAYLOAD, hint="t"))
    assert published.repaired is True
    assert not final.is_symlink()
    assert store.read_bytes(uri) == PAYLOAD
    assert secret.read_bytes() == PAYLOAD  # the symlink target was not modified


SHA = _sha(b"x")


def test_uri_round_trip(store: LocalArtifactStore) -> None:
    uri = store.uri_for(SHA)
    assert uri == f"artifact://sha256/{SHA[:2]}/{SHA}.parquet"
    assert artifact_uri_for(SHA) == uri
    assert sha256_from_uri(uri) == SHA
    assert store.path_for_uri(uri) == store.root / "sha256" / SHA[:2] / f"{SHA}.parquet"


@pytest.mark.parametrize(
    "uri",
    [
        "",
        SHA,
        f"artifact://sha256/{SHA}.parquet",  # missing prefix directory
        f"artifact://sha256/zz/{SHA}.parquet",  # prefix does not match digest
        f"artifact://sha256/{SHA[:2]}/{SHA.upper()}.parquet",  # upper case
        f"artifact://sha256/{SHA[:2]}/{SHA[:-1]}.parquet",  # 63 hex digits
        f"artifact://sha256/{SHA[:2]}/{SHA}0.parquet",  # 65 hex digits
        f"artifact://sha256/{SHA[:2]}/{SHA}.csv",
        f"artifact://sha256/{SHA[:2]}/{SHA}.parquet\n",  # trailing newline
        f" artifact://sha256/{SHA[:2]}/{SHA}.parquet",
        f"artifact://sha256/{SHA[:2]}/{SHA}.parquet/..",
        f"artifact://sha256/{SHA[:2]}/../../../etc/{SHA}.parquet",
        "artifact://sha256/../../etc/passwd",
        f"artifact://tmp/{SHA}.partial",
        f"ARTIFACT://sha256/{SHA[:2]}/{SHA}.parquet",
        f"file:///etc/{SHA}.parquet",
        f"s3://bucket/sha256/{SHA[:2]}/{SHA}.parquet",
        f"/abs/sha256/{SHA[:2]}/{SHA}.parquet",
        f"sha256/{SHA[:2]}/{SHA}.parquet",
        f"artifact://sha256//{SHA}.parquet",
        f"artifact://sha256/{SHA[:2]}/{SHA}.parquet?x=1",
    ],
)
def test_invalid_uris_are_rejected_everywhere(store: LocalArtifactStore, uri: str) -> None:
    with pytest.raises(ValueError, match="invalid artifact URI"):
        store.path_for_uri(uri)
    with pytest.raises(ValueError, match="invalid artifact URI"):
        store.exists(uri)
    with pytest.raises(ValueError, match="invalid artifact URI"):
        store.read_bytes(uri)
    with pytest.raises(ValueError, match="invalid artifact URI"):
        store.verify(uri, SHA)


@pytest.mark.parametrize("value", [None, 42, b"artifact://sha256/aa/x.parquet"])
def test_non_string_uris_are_rejected(store: LocalArtifactStore, value: Any) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        store.path_for_uri(value)


@pytest.mark.parametrize("sha", ["", "abc", SHA.upper(), SHA + "0", "g" * 64, None])
def test_uri_for_rejects_malformed_digest(store: LocalArtifactStore, sha: Any) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        store.uri_for(sha)


def test_read_bytes_verify_and_exists(store: LocalArtifactStore) -> None:
    published = store.commit(store.stage(PAYLOAD, hint="t"))
    sha = published.sha256

    assert store.exists(published.uri)
    assert store.read_bytes(published.uri) == PAYLOAD
    assert store.verify(published.uri, sha)
    assert store.verify(published.uri, sha, expected_size=len(PAYLOAD))
    assert not store.verify(published.uri, sha, expected_size=len(PAYLOAD) + 1)
    assert not store.verify(published.uri, _sha(b"something else"))  # URI names another sha
    with pytest.raises(ValueError, match="SHA-256"):
        store.verify(published.uri, "not-a-sha")


def test_missing_artifact(store: LocalArtifactStore) -> None:
    uri = store.uri_for(SHA)
    assert not store.exists(uri)
    assert not store.verify(uri, SHA)
    with pytest.raises(FileNotFoundError):
        store.read_bytes(uri)


def test_read_bytes_detects_corruption(store: LocalArtifactStore) -> None:
    published = store.commit(store.stage(PAYLOAD, hint="t"))
    _overwrite(store.path_for_uri(published.uri), b"tampered")

    assert store.exists(published.uri)  # exists() does not hash
    with pytest.raises(ArtifactIntegrityError, match="corrupt"):
        store.read_bytes(published.uri)
    assert store.read_bytes(published.uri, verify=False) == b"tampered"


def test_iter_artifacts_lists_only_published_artifacts(store: LocalArtifactStore) -> None:
    published = [store.commit(store.stage(f"result {i}".encode(), hint="t")) for i in range(3)]
    store.stage(b"staged but never committed", hint="t")
    objects = store.root / "sha256"
    (objects / "zz").mkdir()
    (objects / "zz" / "notes.txt").write_text("stray")
    wrong_dir_sha = _sha(b"misplaced")
    wrong_prefix = "00" if wrong_dir_sha[:2] != "00" else "11"
    (objects / wrong_prefix).mkdir(exist_ok=True)
    (objects / wrong_prefix / f"{wrong_dir_sha}.parquet").write_bytes(b"misplaced")
    (objects / "README").write_text("stray file at the top level")

    listed = list(store.iter_artifacts())

    assert listed == sorted((p.uri, store.path_for_uri(p.uri)) for p in published)


def test_iter_artifacts_on_empty_store(store: LocalArtifactStore) -> None:
    assert list(store.iter_artifacts()) == []


def test_rename_fallback_when_hard_links_are_unsupported(
    store: LocalArtifactStore, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[tuple[Any, Any]] = []

    def no_links(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        calls.append((src, dst))
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", no_links)
    with caplog.at_level(logging.INFO, logger="quantgrid.storage.artifacts"):
        first = store.commit(store.stage(PAYLOAD, hint="a"))
        second = store.commit(store.stage(PAYLOAD, hint="b"))
        third = store.commit(store.stage(b"other", hint="c"))

    assert (first.created, second.created, third.created) == (True, False, True)
    assert len(calls) == 1  # the store remembers that links are unsupported
    assert store.read_bytes(first.uri) == PAYLOAD
    assert _temp_files(store) == []
    assert any("os.replace" in r.getMessage() for r in caplog.records)


def test_unexpected_link_error_is_reported_as_artifact_error(
    store: LocalArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "link", broken)
    staged = store.stage(PAYLOAD, hint="t")
    with pytest.raises(ArtifactError, match="failed to publish"):
        store.commit(staged)
    assert _object_files(store) == []


def test_local_store_satisfies_protocol(store: LocalArtifactStore) -> None:
    assert isinstance(store, ArtifactStore)


def test_from_settings_uses_artifact_root(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "configured")
    store = LocalArtifactStore.from_settings(settings)
    assert store.root == (tmp_path / "configured").resolve()
    assert "configured" in repr(store)


N_PROCESSES = 6
PAYLOADS_PER_PROCESS = 4


def _payload(k: int) -> bytes:
    return f"deterministic result #{k}\n".encode() * 2048


def _commit_concurrently(root: str, worker: int, barrier: MPBarrier, results: MPQueue[Any]) -> None:
    """Child process: stage each payload, wait for every process, then commit at once."""
    try:
        store = LocalArtifactStore(Path(root))
        out = []
        for k in range(PAYLOADS_PER_PROCESS):
            staged = store.stage(_payload(k), hint=f"w{worker}-p{k}")
            barrier.wait(timeout=60)
            published = store.commit(staged)
            out.append((k, published.uri, published.sha256, published.size, published.created))
        results.put((worker, out, None))
    except BaseException as exc:
        barrier.abort()
        results.put((worker, [], repr(exc)))


def test_concurrent_commits_from_many_processes(store: LocalArtifactStore) -> None:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(N_PROCESSES)
    results: MPQueue[Any] = ctx.Queue()
    procs = [
        ctx.Process(target=_commit_concurrently, args=(str(store.root), i, barrier, results))
        for i in range(N_PROCESSES)
    ]
    for proc in procs:
        proc.start()
    collected = [results.get(timeout=120) for _ in procs]
    for proc in procs:
        proc.join(timeout=60)

    assert [p.exitcode for p in procs] == [0] * N_PROCESSES
    assert [err for _, _, err in collected if err] == []
    by_payload: dict[int, list[tuple[str, str, int, bool]]] = {}
    for _, out, _ in collected:
        for k, uri, sha, size, created in out:
            by_payload.setdefault(k, []).append((uri, sha, size, created))

    for k in range(PAYLOADS_PER_PROCESS):
        outcomes = by_payload[k]
        expected_sha = _sha(_payload(k))
        assert len(outcomes) == N_PROCESSES  # every commit succeeded
        assert {(uri, sha, size) for uri, sha, size, _ in outcomes} == {
            (artifact_uri_for(expected_sha), expected_sha, len(_payload(k)))
        }
        assert sum(created for *_, created in outcomes) == 1  # exactly one creator
        assert store.verify(artifact_uri_for(expected_sha), expected_sha)
    assert len(_object_files(store)) == PAYLOADS_PER_PROCESS  # exactly one file each
    assert _temp_files(store) == []  # no leftover temp files


def test_concurrent_commits_from_many_threads(store: LocalArtifactStore) -> None:
    n_threads, rounds = 12, 25
    barrier = threading.Barrier(n_threads)
    results: list[list[PublishedArtifact]] = [[] for _ in range(n_threads)]
    errors: list[BaseException] = []

    def run(worker: int) -> None:
        try:
            for r in range(rounds):
                staged = store.stage(_payload(r), hint=f"t{worker}")
                barrier.wait(timeout=30)
                results[worker].append(store.commit(staged))
        except BaseException as exc:
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert errors == []
    for r in range(rounds):
        outcomes = [results[w][r] for w in range(n_threads)]
        assert {o.sha256 for o in outcomes} == {_sha(_payload(r))}
        assert sum(o.created for o in outcomes) == 1
    assert len(_object_files(store)) == rounds
    assert _temp_files(store) == []
