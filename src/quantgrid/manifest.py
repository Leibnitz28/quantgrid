"""Reproducibility manifest: exact code, environment, dependency and dataset identity.

* ``code_sha``: SHA-256 over every ``*.py`` of the installed package (path + bytes,
  sorted); exact for uncommitted changes.
* ``git_commit``: informational, plus a dirty flag.
* ``env_sha``: SHA-256 over interpreter, platform and sorted installed distributions.
* ``lockfile_sha``: SHA-256 of ``uv.lock`` when present.

Batches store the admitting process's manifest; attempts record ``code_sha``/``env_sha``,
and cached results are reused only when both match.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

import quantgrid
from quantgrid.domain.hashing import canonical_sha256, sha256_hex
from quantgrid.domain.specs import RESULT_SCHEMA_VERSION, DatasetInfo

MANIFEST_VERSION = 1
PACKAGE_ROOT = Path(quantgrid.__file__).resolve().parent


@lru_cache(maxsize=1)
def code_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _repo_root() -> Path | None:
    for parent in (PACKAGE_ROOT, *PACKAGE_ROOT.parents):
        if (parent / ".git").exists() or (parent / "uv.lock").exists():
            return parent
    return None


@lru_cache(maxsize=1)
def git_info() -> dict[str, Any]:
    env_commit = os.environ.get("QG_GIT_COMMIT")
    if env_commit:
        return {"git_commit": env_commit, "git_dirty": None, "source": "env"}
    root = _repo_root()
    if root is None or not (root / ".git").exists():
        return {"git_commit": None, "git_dirty": None, "source": "unavailable"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - fixed argv, informational only
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None, "source": "error"}
    return {"git_commit": commit or None, "git_dirty": dirty, "source": "git"}


@lru_cache(maxsize=1)
def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = dist.version
    return dict(sorted(packages.items()))


@lru_cache(maxsize=1)
def environment_manifest() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": installed_packages(),
    }
    # Excludes the kernel string (varies per host patch level); includes everything that
    # can change numerical results.
    hashed = {k: v for k, v in env.items() if k != "platform"}
    env["env_sha"] = canonical_sha256(hashed)
    root = _repo_root()
    lock = root / "uv.lock" if root else None
    env["lockfile_sha"] = sha256_hex(lock.read_bytes()) if lock and lock.exists() else None
    return env


def env_sha() -> str:
    return str(environment_manifest()["env_sha"])


def build_manifest(dataset: DatasetInfo, spec_sha: str) -> dict[str, Any]:
    """Manifest stored on a batch at admission."""
    env = environment_manifest()
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code": {
            "code_sha": code_digest(),
            "package_version": quantgrid.__version__,
            **git_info(),
        },
        "environment": env,
        "dataset": dataset.model_dump(mode="json"),
        "spec_sha": spec_sha,
        "result_schema_version": RESULT_SCHEMA_VERSION,
    }


def manifest_drift(recorded: dict[str, Any]) -> dict[str, Any]:
    """Differences between a recorded manifest and the current process."""
    current_env = environment_manifest()
    rec_env = recorded.get("environment", {})
    rec_pkgs: dict[str, str] = rec_env.get("packages", {})
    cur_pkgs = current_env["packages"]
    changed = {
        name: {"recorded": rec_pkgs.get(name), "current": cur_pkgs.get(name)}
        for name in sorted(set(rec_pkgs) | set(cur_pkgs))
        if rec_pkgs.get(name) != cur_pkgs.get(name)
    }
    rec_code = recorded.get("code", {}).get("code_sha")
    return {
        "code_sha_match": rec_code == code_digest(),
        "recorded_code_sha": rec_code,
        "current_code_sha": code_digest(),
        "env_sha_match": rec_env.get("env_sha") == current_env["env_sha"],
        "python": {"recorded": rec_env.get("python"), "current": current_env["python"]},
        "machine": {"recorded": rec_env.get("machine"), "current": current_env["machine"]},
        "changed_packages": changed,
    }
