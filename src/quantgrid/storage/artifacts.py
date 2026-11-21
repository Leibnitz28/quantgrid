"""Content-addressed, crash-safe storage for task result artifacts.

Layout under the store root::

    tmp/<hint>.<uuid4 hex>.partial       staged bytes, not yet published
    sha256/<aa>/<sha256>.parquet         published artifacts (aa = sha256[:2])

The only accepted artifact URI is ``artifact://sha256/<aa>/<sha256>.parquet``. Staged files
use informational ``artifact://tmp/<name>`` URIs that :meth:`path_for_uri` never resolves.

Publishing is stage (write, hash, fsync a temp file) then commit (re-hash, no-clobber hard
link to the content address, fsync the directory; ``os.replace`` if hard links are
unsupported). Invariants:

* A published file only appears by linking or renaming a complete, fsynced, re-verified
  temp file, so readers never see a partial artifact.
* Different bytes have different addresses, so a result never overwrites another. An
  existing file is replaced only when it fails verification, and only with the exact bytes
  its name promises.
* Commit is idempotent, also under concurrent writers: one file remains and no temp files.
* A crash between stage and commit leaves only a ``.partial`` file; :meth:`gc_temp`
  removes it. Files are created read-only (0o444); reads never follow symlinks.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import math
import os
import re
import stat
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from quantgrid.config import Settings, get_settings
from quantgrid.domain.errors import ArtifactError
from quantgrid.domain.hashing import is_sha256_hex

logger = logging.getLogger(__name__)

URI_PREFIX = "artifact://sha256/"
TEMP_URI_PREFIX = "artifact://tmp/"
ARTIFACT_SUFFIX = ".parquet"
TEMP_SUFFIX = ".partial"
TMP_DIRNAME = "tmp"
OBJECTS_DIRNAME = "sha256"
#: Permission bits of staged and published files (immutable content).
FILE_MODE = 0o444

_CHUNK = 1 << 20
_MAX_HINT_LEN = 64
_URI_RE = re.compile(r"artifact://sha256/(?P<prefix>[0-9a-f]{2})/(?P<sha>[0-9a-f]{64})\.parquet")
_OBJECT_NAME_RE = re.compile(r"(?P<sha>[0-9a-f]{64})\.parquet")
_PREFIX_RE = re.compile(r"[0-9a-f]{2}")
_HINT_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")

_O_NOFOLLOW: int = getattr(os, "O_NOFOLLOW", 0)
_O_NONBLOCK: int = getattr(os, "O_NONBLOCK", 0)
_O_CLOEXEC: int = getattr(os, "O_CLOEXEC", 0)
_O_DIRECTORY: int = getattr(os, "O_DIRECTORY", 0)
#: ``os.link`` errors meaning "this filesystem cannot hard-link". We then fall back to
#: ``os.replace``.
_LINK_UNSUPPORTED_ERRNOS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})
#: ``fsync`` on a directory is not supported everywhere. These errors are ignored.
_DIR_FSYNC_UNSUPPORTED_ERRNOS = frozenset({errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP})
#: Digest placeholder for a path that exists but cannot be verified (symlink, directory,
#: unreadable). It never equals a real ``(sha256, size)`` pair.
_UNVERIFIABLE: tuple[str, int] = ("", -1)


class ArtifactIntegrityError(ArtifactError):
    """Bytes do not match the digest they should have."""


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    """Bytes written to a temp file, not yet published; ``commit`` re-checks digest and size."""

    temp_path: Path
    temp_uri: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """Result of a commit.

    ``created`` is True if this call put the bytes at ``uri`` (fresh publish or ``repaired``
    corrupt file) and False if a verified copy was already there.
    """

    uri: str
    sha256: str
    size: int
    created: bool
    repaired: bool = False


@runtime_checkable
class ArtifactStore(Protocol):
    """Content-addressed artifact storage; implementations keep the module invariants.

    ``path_for_uri`` and ``iter_artifacts`` expose local paths; a remote store may not.
    """

    def uri_for(self, sha256: str) -> str: ...

    def path_for_uri(self, uri: str) -> Path: ...

    def stage(self, data: bytes, *, hint: str) -> StagedArtifact: ...

    def commit(self, staged: StagedArtifact) -> PublishedArtifact: ...

    def discard(self, staged: StagedArtifact) -> None: ...

    def read_bytes(self, uri: str, *, verify: bool = True) -> bytes: ...

    def verify(self, uri: str, expected_sha: str, *, expected_size: int | None = None) -> bool: ...

    def exists(self, uri: str) -> bool: ...

    def gc_temp(self, older_than_s: float) -> int: ...

    def iter_artifacts(self) -> Iterator[tuple[str, Path]]: ...


def _preview(value: object, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...'"


def artifact_uri_for(sha256: str) -> str:
    if not is_sha256_hex(sha256):
        raise ValueError(f"not a lowercase hex SHA-256 digest: {_preview(sha256)}")
    return f"{URI_PREFIX}{sha256[:2]}/{sha256}{ARTIFACT_SUFFIX}"


def sha256_from_uri(uri: str) -> str:
    """Digest in an artifact URI. Only ``artifact://sha256/<aa>/<64 lowercase hex>.parquet`` with
    ``aa == sha[:2]`` is accepted; anything else raises ``ValueError``.
    """
    if not isinstance(uri, str):
        raise ValueError(f"artifact URI must be a string, got {type(uri).__name__}")
    match = _URI_RE.fullmatch(uri)
    if match is None or match["prefix"] != match["sha"][:2]:
        raise ValueError(
            f"invalid artifact URI {_preview(uri)}: expected "
            "'artifact://sha256/<aa>/<64 lowercase hex>.parquet'"
        )
    return match["sha"]


def _sanitise_hint(hint: str) -> str:
    cleaned = _HINT_UNSAFE_RE.sub("_", str(hint))[:_MAX_HINT_LEN].lstrip(".-")
    return cleaned or "artifact"


class _NotARegularFile(OSError):
    """A symlink, directory, FIFO or other non-regular file."""


def _open_regular(path: Path) -> int:
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC)
    except OSError as exc:
        if exc.errno == errno.ELOOP:  # O_NOFOLLOW refused a symlink
            raise _NotARegularFile(errno.ELOOP, "artifact path is a symlink", str(path)) from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _NotARegularFile(errno.EINVAL, "artifact path is not a regular file", str(path))
    except BaseException:
        os.close(fd)
        raise
    return fd


def _file_digest(path: Path) -> tuple[str, int]:
    fd = _open_regular(path)
    digest = hashlib.sha256()
    size = 0
    buffer = bytearray(_CHUNK)
    view = memoryview(buffer)
    with os.fdopen(fd, "rb", buffering=0) as fh:
        while True:
            n = fh.readinto(view)
            if not n:
                break
            digest.update(view[:n])
            size += n
    return digest.hexdigest(), size


def _probe_digest(path: Path) -> tuple[str, int] | None:
    """``(sha256, size)`` of ``path``; ``None`` if absent, ``_UNVERIFIABLE`` if unverifiable."""
    try:
        return _file_digest(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return _UNVERIFIABLE


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - directories cannot be opened on Windows
        return
    fd = os.open(path, os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC)
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _DIR_FSYNC_UNSUPPORTED_ERRNOS:
            raise
    finally:
        os.close(fd)


def _ensure_dir(path: Path) -> None:
    """Create ``path`` and missing parents, fsyncing each new directory's parent."""
    if path.is_dir():
        return
    missing: list[Path] = []
    current = path
    while not current.is_dir():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        # FileExistsError: created concurrently (or it is a file, which the next
        # operation reports).
        with contextlib.suppress(FileExistsError):
            directory.mkdir()
        _fsync_dir(directory.parent)


def _unlink_quietly(path: Path) -> None:
    """Best-effort temp file removal; failures are logged and left for ``gc_temp``."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("could not remove temp artifact file %s: %s", path, exc)


class LocalArtifactStore:
    """:class:`ArtifactStore` on a local or shared POSIX filesystem.

    ``root/tmp`` and ``root/sha256`` must share a filesystem so link and rename are atomic.
    Instances hold no locks; separate threads and processes can share one root.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._tmp_dir = self._root / TMP_DIRNAME
        self._objects_dir = self._root / OBJECTS_DIRNAME
        # Flipped to False the first time the filesystem refuses hard links.
        self._use_hardlinks = True

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LocalArtifactStore:
        return cls((settings or get_settings()).artifact_root)

    def __repr__(self) -> str:
        return f"LocalArtifactStore(root={str(self._root)!r})"

    @property
    def root(self) -> Path:
        return self._root

    def uri_for(self, sha256: str) -> str:
        return artifact_uri_for(sha256)

    def path_for_uri(self, uri: str) -> Path:
        """Local path of an artifact URI; ``ValueError`` unless it has the exact form."""
        return self._object_path(sha256_from_uri(uri))

    def _object_path(self, sha256: str) -> Path:
        return self._objects_dir / sha256[:2] / f"{sha256}{ARTIFACT_SUFFIX}"

    def stage(self, data: bytes, *, hint: str) -> StagedArtifact:
        """Write ``data`` to ``tmp/<hint>.<uuid4 hex>.partial``, hashing while writing, then fsync.

        On failure the temp file is removed and :class:`ArtifactError` is raised.
        """
        if not isinstance(data, bytes | bytearray | memoryview):
            raise TypeError(f"artifact data must be bytes, got {type(data).__name__}")
        name = f"{_sanitise_hint(hint)}.{uuid.uuid4().hex}{TEMP_SUFFIX}"
        path = self._tmp_dir / name
        digest = hashlib.sha256()
        size = 0
        try:
            _ensure_dir(self._tmp_dir)
            fd = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC, FILE_MODE
            )
        except OSError as exc:
            raise ArtifactError(f"cannot create temp artifact file {path}: {exc}") from exc
        try:
            with os.fdopen(fd, "wb") as fh:
                view = memoryview(data).cast("B")
                for start in range(0, len(view), _CHUNK):
                    chunk = view[start : start + _CHUNK]
                    digest.update(chunk)
                    fh.write(chunk)
                    size += len(chunk)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException as exc:
            _unlink_quietly(path)
            if isinstance(exc, OSError):
                raise ArtifactError(f"failed to stage artifact {path}: {exc}") from exc
            raise
        return StagedArtifact(
            temp_path=path,
            temp_uri=f"{TEMP_URI_PREFIX}{name}",
            sha256=digest.hexdigest(),
            size=size,
        )

    def commit(self, staged: StagedArtifact) -> PublishedArtifact:
        """Publish a staged artifact at its content address (atomic, idempotent).

        If the address is free, publish (``created=True``). If a verified copy exists, drop the temp
        file (``created=False``). If the existing file fails verification, replace it with the temp
        file and log a warning (``created=True, repaired=True``). The temp file is re-hashed first
        and deleted with :class:`ArtifactIntegrityError` if it changed; this does not stop a hostile
        same-user process writing between the check and the link.
        """
        self._check_staged(staged)
        sha, size = staged.sha256, staged.size
        final = self._object_path(sha)
        uri = artifact_uri_for(sha)

        try:
            temp_digest = _file_digest(staged.temp_path)
        except FileNotFoundError:
            return self._commit_without_temp(staged, final, uri)
        except OSError as exc:
            raise ArtifactError(f"cannot read staged artifact {staged.temp_uri}: {exc}") from exc
        if temp_digest != (sha, size):
            _unlink_quietly(staged.temp_path)
            raise ArtifactIntegrityError(
                f"staged artifact {staged.temp_uri} changed after staging: staged "
                f"sha256={sha} size={size}, now sha256={temp_digest[0]} size={temp_digest[1]}"
            )

        try:
            _ensure_dir(final.parent)
            if self._publish_no_clobber(staged.temp_path, final):
                _unlink_quietly(staged.temp_path)  # no-op after the rename fallback
                return PublishedArtifact(uri=uri, sha256=sha, size=size, created=True)

            existing = _probe_digest(final)
            if existing == (sha, size):
                _unlink_quietly(staged.temp_path)
                return PublishedArtifact(uri=uri, sha256=sha, size=size, created=False)

            # Corrupt or unreadable (or deleted concurrently): replace it with the verified
            # temp file. This is always safe, because the temp file holds exactly the
            # bytes this address promises.
            if existing is not None:
                found_sha, found_size = existing
                logger.warning(
                    "replacing corrupt artifact %s with a verified copy "
                    "(expected sha256=%s size=%d, found sha256=%s size=%d)",
                    uri,
                    sha,
                    size,
                    found_sha or "<unreadable>",
                    found_size,
                )
            os.replace(staged.temp_path, final)
            _fsync_dir(final.parent)
            return PublishedArtifact(
                uri=uri, sha256=sha, size=size, created=True, repaired=existing is not None
            )
        except OSError as exc:
            raise ArtifactError(f"failed to publish artifact {uri}: {exc}") from exc

    def discard(self, staged: StagedArtifact) -> None:
        """Remove a staged temp file (idempotent). Never touches a published artifact."""
        self._check_staged(staged)
        _unlink_quietly(staged.temp_path)

    def _check_staged(self, staged: StagedArtifact) -> None:
        if not isinstance(staged, StagedArtifact):
            raise TypeError(f"expected StagedArtifact, got {type(staged).__name__}")
        path = staged.temp_path
        if path.parent != self._tmp_dir or not path.name.endswith(TEMP_SUFFIX):
            raise ValueError(f"staged artifact {path} does not belong to {self!r}")
        if not is_sha256_hex(staged.sha256) or staged.size < 0:
            raise ValueError(f"malformed staged artifact: {staged!r}")

    def _publish_no_clobber(self, temp: Path, final: Path) -> bool:
        """Link ``temp`` to ``final`` if nothing is there; return False if ``final`` exists.

        With hard links this is an atomic put-if-absent. The rename fallback can let two committers
        both win, which is harmless because both temp files hold the same verified bytes.
        """
        if self._use_hardlinks:
            try:
                os.link(temp, final)
            except FileExistsError:
                return False
            except OSError as exc:
                if exc.errno not in _LINK_UNSUPPORTED_ERRNOS:
                    raise
                logger.info(
                    "filesystem under %s refused a hard link (%s); publishing with "
                    "os.replace from now on",
                    self._root,
                    exc,
                )
                self._use_hardlinks = False
            else:
                _fsync_dir(final.parent)
                return True
        if os.path.lexists(final):
            return False
        os.replace(temp, final)
        _fsync_dir(final.parent)
        return True

    def _commit_without_temp(
        self, staged: StagedArtifact, final: Path, uri: str
    ) -> PublishedArtifact:
        """The temp file is gone; succeed only if a verified copy is published."""
        if _probe_digest(final) == (staged.sha256, staged.size):
            return PublishedArtifact(uri=uri, sha256=staged.sha256, size=staged.size, created=False)
        raise ArtifactError(
            f"staged artifact {staged.temp_uri} no longer exists (discarded or "
            f"garbage-collected) and no verified copy is published at {uri}"
        )

    def read_bytes(self, uri: str, *, verify: bool = True) -> bytes:
        """Bytes of a published artifact. With ``verify`` (default) corrupt content raises
        :class:`ArtifactIntegrityError`. Raises ``FileNotFoundError``, ``OSError`` for a symlink or
        non-regular file, ``ValueError`` for an invalid URI.
        """
        sha = sha256_from_uri(uri)
        fd = _open_regular(self._object_path(sha))
        with os.fdopen(fd, "rb") as fh:
            data = fh.read()
        if verify and hashlib.sha256(data).hexdigest() != sha:
            raise ArtifactIntegrityError(
                f"artifact {uri} is corrupt: its content does not hash to its address"
            )
        return data

    def verify(self, uri: str, expected_sha: str, *, expected_size: int | None = None) -> bool:
        """True only if ``uri`` names ``expected_sha`` and the stored bytes match it (and size)."""
        sha = sha256_from_uri(uri)
        if not is_sha256_hex(expected_sha):
            raise ValueError(f"not a lowercase hex SHA-256 digest: {_preview(expected_sha)}")
        if sha != expected_sha:
            return False
        digest = _probe_digest(self._object_path(sha))
        if digest is None or digest[0] != sha:
            return False
        return expected_size is None or digest[1] == expected_size

    def exists(self, uri: str) -> bool:
        path = self.path_for_uri(uri)
        try:
            return stat.S_ISREG(os.lstat(path).st_mode)
        except (FileNotFoundError, NotADirectoryError):
            return False

    def gc_temp(self, older_than_s: float) -> int:
        """Delete ``*.partial`` files older than ``older_than_s`` seconds; return the count.

        Use a threshold well above a live worker's stage-to-commit gap; removing its temp file
        makes that commit fail with a retryable :class:`ArtifactError`.
        """
        if not math.isfinite(older_than_s) or older_than_s < 0:
            raise ValueError(f"older_than_s must be a finite number >= 0, got {older_than_s!r}")
        cutoff = time.time() - older_than_s
        removed = 0
        try:
            entries = os.scandir(self._tmp_dir)
        except FileNotFoundError:
            return 0
        with entries:
            for entry in entries:
                if not entry.name.endswith(TEMP_SUFFIX):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                    if not stat.S_ISREG(st.st_mode) or st.st_mtime >= cutoff:
                        continue
                    os.unlink(entry.path)
                except FileNotFoundError:
                    continue  # committed or removed concurrently
                except OSError as exc:  # keep sweeping; report what could not be removed
                    logger.warning("could not remove stale temp file %s: %s", entry.path, exc)
                    continue
                removed += 1
        if removed:
            logger.info("removed %d stale temp artifact file(s) from %s", removed, self._tmp_dir)
        return removed

    def iter_artifacts(self) -> Iterator[tuple[str, Path]]:
        """Yield ``(uri, path)`` for every published artifact, sorted by URI.

        Strays, symlinks and temp files are skipped. The verifier uses this to find orphans.
        """
        try:
            with os.scandir(self._objects_dir) as it:
                prefixes = sorted(
                    e.name
                    for e in it
                    if _PREFIX_RE.fullmatch(e.name) and e.is_dir(follow_symlinks=False)
                )
        except FileNotFoundError:
            return
        for prefix in prefixes:
            directory = self._objects_dir / prefix
            try:
                with os.scandir(directory) as it:
                    names = sorted(e.name for e in it if e.is_file(follow_symlinks=False))
            except FileNotFoundError:
                continue
            for name in names:
                match = _OBJECT_NAME_RE.fullmatch(name)
                if match is not None and match["sha"][:2] == prefix:
                    yield artifact_uri_for(match["sha"]), directory / name
