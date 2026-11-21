"""Content-addressed, crash-safe storage for task result artifacts.

Layout under the store root::

    tmp/<hint>.<uuid4 hex>.partial       staged bytes, not yet published
    sha256/<aa>/<sha256>.parquet         published artifacts (aa = sha256[:2])

The only accepted artifact URI form is ``artifact://sha256/<aa>/<sha256>.parquet``
(64 lowercase hex digits, ``aa`` equal to the first two). Staged files are described
by ``artifact://tmp/<name>`` URIs, which are informational: they are recorded in
``result_artifacts.temp_uri`` and are never resolvable through :meth:`path_for_uri`.

Publishing is "temp file + atomic rename / staged object commit":

1. :meth:`LocalArtifactStore.stage` writes the bytes to a fresh temp file
   (``O_CREAT | O_EXCL``), hashes them while writing, then flushes and fsyncs.
2. :meth:`LocalArtifactStore.commit` re-hashes the temp file, checks it against the
   staged digest and size, and publishes it at its content address with a no-clobber
   hard link (``os.link``), then fsyncs the parent directory. If the filesystem cannot
   hard-link, it falls back to ``os.replace``. The temp name is then removed.

Invariants:

* A file at ``sha256/<aa>/<sha>.parquet`` only appears by atomically linking or renaming
  a complete, fsynced and re-verified temp file whose SHA-256 is ``<sha>``. Readers never
  see a partially written artifact.
* A different result can never overwrite an existing artifact, because different bytes
  have a different address. An existing file is only replaced when it fails verification
  (corruption), and only with a verified copy of exactly the bytes its name promises.
* Commit is idempotent. Committing the same bytes again (another attempt, or a "zombie"
  worker racing its replacement) returns ``created=False`` and leaves the published file
  alone. N processes committing identical bytes concurrently all succeed, exactly one
  file remains and no temp files are left. Where hard links are supported, exactly one
  of them reports ``created=True``.
* A crash between stage and commit leaves only a ``.partial`` temp file and never a
  partial artifact. :meth:`LocalArtifactStore.gc_temp` removes such leftovers.
* Published files and temp files are created read-only (``0o444``), as git does for its
  object store. Reads never follow symlinks and only accept regular files.
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
    """Stored or staged bytes do not match the digest they are supposed to have."""


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    """Bytes durably written to a temp file, not yet published.

    ``sha256``/``size`` describe the bytes as they were written. :meth:`ArtifactStore.commit`
    re-checks them before publishing.
    """

    temp_path: Path
    temp_uri: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    """Result of a commit.

    ``created`` is True if this call put the bytes at ``uri``: a fresh publish, or the
    repair of a corrupt file (``repaired=True``). It is False if a verified copy was
    already there (idempotent duplicate).
    """

    uri: str
    sha256: str
    size: int
    created: bool
    repaired: bool = False


@runtime_checkable
class ArtifactStore(Protocol):
    """Content-addressed artifact storage (local filesystem now, S3-compatible later).

    Implementations must keep the invariants in the module docstring: atomic,
    idempotent, no-clobber publication keyed by SHA-256, and strict URI validation.
    ``path_for_uri`` and ``iter_artifacts`` expose local paths. A remote store would
    return local cache paths there, or raise ``NotImplementedError``.
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


# --------------------------------------------------------------------------------------
# URIs
# --------------------------------------------------------------------------------------


def _preview(value: object, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...'"


def artifact_uri_for(sha256: str) -> str:
    """``artifact://sha256/<aa>/<sha256>.parquet`` for a lowercase hex SHA-256 digest."""
    if not is_sha256_hex(sha256):
        raise ValueError(f"not a lowercase hex SHA-256 digest: {_preview(sha256)}")
    return f"{URI_PREFIX}{sha256[:2]}/{sha256}{ARTIFACT_SUFFIX}"


def sha256_from_uri(uri: str) -> str:
    """Digest encoded in an artifact URI.

    Only the exact form ``artifact://sha256/<aa>/<64 lowercase hex>.parquet`` with
    ``aa == sha[:2]`` is accepted. Other schemes, path traversal, extra or missing
    segments, upper case, whitespace and non-strings all raise ``ValueError``.
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
    """A safe file-name fragment. It can never contain a separator or start with '.'/'-'."""
    cleaned = _HINT_UNSAFE_RE.sub("_", str(hint))[:_MAX_HINT_LEN].lstrip(".-")
    return cleaned or "artifact"


# --------------------------------------------------------------------------------------
# Low-level file helpers
# --------------------------------------------------------------------------------------


class _NotARegularFile(OSError):
    """The path exists but is a symlink, directory, FIFO or other non-regular file."""


def _open_regular(path: Path) -> int:
    """Open ``path`` read-only without following symlinks. Returns a regular file's fd."""
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
    """Streamed ``(sha256 hex, size)`` of a regular file (constant memory)."""
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
    """``(sha256, size)`` of ``path``. ``None`` if it does not exist, ``_UNVERIFIABLE`` if
    it exists but cannot be verified (symlink, non-regular file, I/O or permission error)."""
    try:
        return _file_digest(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return _UNVERIFIABLE


def _fsync_dir(path: Path) -> None:
    """Make directory-entry changes (create/rename/link) in ``path`` durable."""
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
    """Create ``path`` and any missing parents, fsyncing each new directory's parent.
    Safe when several processes create the same directory at once."""
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
    """Best-effort removal of a temp file. Never raises; failures are logged and left for
    :meth:`LocalArtifactStore.gc_temp`."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("could not remove temp artifact file %s: %s", path, exc)


# --------------------------------------------------------------------------------------
# Local filesystem store
# --------------------------------------------------------------------------------------


class LocalArtifactStore:
    """:class:`ArtifactStore` on a local or shared POSIX filesystem.

    ``root/tmp`` and ``root/sha256`` must be on the same filesystem, so that link and
    rename are atomic. Instances hold no locks and can be shared between threads.
    Separate processes, even on different hosts with a shared mount, can use the same
    root at the same time.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()
        self._tmp_dir = self._root / TMP_DIRNAME
        self._objects_dir = self._root / OBJECTS_DIRNAME
        # Flipped to False the first time the filesystem refuses hard links.
        self._use_hardlinks = True

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> LocalArtifactStore:
        """Store rooted at ``QG_ARTIFACT_ROOT``. A relative path is resolved against the
        current working directory."""
        return cls((settings or get_settings()).artifact_root)

    def __repr__(self) -> str:
        return f"LocalArtifactStore(root={str(self._root)!r})"

    @property
    def root(self) -> Path:
        return self._root

    # ---- addressing -------------------------------------------------------------------

    def uri_for(self, sha256: str) -> str:
        return artifact_uri_for(sha256)

    def path_for_uri(self, uri: str) -> Path:
        """Local path of an artifact URI. Raises ``ValueError`` for anything but the exact
        ``artifact://sha256/<aa>/<sha>.parquet`` form, so it cannot leave the store root."""
        return self._object_path(sha256_from_uri(uri))

    def _object_path(self, sha256: str) -> Path:
        return self._objects_dir / sha256[:2] / f"{sha256}{ARTIFACT_SUFFIX}"

    # ---- publishing -------------------------------------------------------------------

    def stage(self, data: bytes, *, hint: str) -> StagedArtifact:
        """Durably write ``data`` to ``tmp/<sanitised hint>.<uuid4 hex>.partial``.

        The digest is computed while writing. The file is flushed and fsynced before
        this returns. On failure the temp file is removed and :class:`ArtifactError`
        is raised.
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

        * no artifact at the address: publish it (``created=True``);
        * a verified copy is already there: drop the temp file (``created=False``);
        * the existing file fails verification: atomically replace it with the verified
          temp file and log a warning (``created=True, repaired=True``).

        The temp file is re-hashed first. If it changed after staging it is deleted and
        :class:`ArtifactIntegrityError` is raised. This catches accidental modification;
        it cannot stop a hostile same-user process writing between the check and the
        link. Committing the same ``staged`` again after a successful commit returns
        ``created=False``. Raises :class:`ArtifactError` on I/O failure.
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
        """Remove a staged temp file (best effort, idempotent). After a commit this does
        nothing; it never touches a published artifact."""
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
        """Publish ``temp`` as ``final`` if nothing is at ``final`` yet.

        Returns True if published, False if ``final`` already existed (caller verifies).
        With hard links this is an atomic put-if-absent. In the rename fallback, two
        concurrent committers can both "win", which is harmless because both temp files
        hold the same verified bytes.
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
        """The temp file is gone: an earlier commit of ``staged`` succeeded, or the temp
        file was discarded or garbage-collected. Succeed only if a verified copy is
        published."""
        if _probe_digest(final) == (staged.sha256, staged.size):
            return PublishedArtifact(uri=uri, sha256=staged.sha256, size=staged.size, created=False)
        raise ArtifactError(
            f"staged artifact {staged.temp_uri} no longer exists (discarded or "
            f"garbage-collected) and no verified copy is published at {uri}"
        )

    # ---- reading ----------------------------------------------------------------------

    def read_bytes(self, uri: str, *, verify: bool = True) -> bytes:
        """Bytes of a published artifact.

        With ``verify=True`` (the default) the content is hashed and
        :class:`ArtifactIntegrityError` is raised if it does not match the URI, so
        corrupt bytes are never returned. Raises ``FileNotFoundError`` if the artifact
        does not exist, ``OSError`` if the path is a symlink or not a regular file
        (never followed), and ``ValueError`` for an invalid URI.
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
        """True only if ``uri`` names ``expected_sha`` and the stored bytes hash to it
        (and have ``expected_size`` bytes, if given). False if the artifact is missing,
        unreadable, not a regular file or corrupt. ``ValueError`` for a malformed URI or
        digest."""
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
        """True if a regular file is published at ``uri``. The content is not verified."""
        path = self.path_for_uri(uri)
        try:
            return stat.S_ISREG(os.lstat(path).st_mode)
        except (FileNotFoundError, NotADirectoryError):
            return False

    # ---- maintenance ------------------------------------------------------------------

    def gc_temp(self, older_than_s: float) -> int:
        """Delete ``*.partial`` temp files last modified more than ``older_than_s`` seconds
        ago. These are left behind by crashes between stage and commit. Returns the
        number deleted.

        Use a threshold well above the longest stage-to-commit gap of a live worker. A
        removed temp file makes that worker's commit fail with :class:`ArtifactError`,
        which is retryable. It never corrupts a published artifact.
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

        Only regular files named ``<sha>.parquet`` inside their own ``<sha[:2]>``
        directory are listed. Stray files, symlinks and temp files are skipped. Used by
        the verifier to find orphans (files with no ``result_artifacts`` row).
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
