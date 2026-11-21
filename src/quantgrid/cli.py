"""``quantgrid`` command: run services and operator tools.

quantgrid migrate                 apply Alembic migrations
quantgrid api [--host --port]     HTTP API
quantgrid coordinator             scheduler loop (one leader, N standbys)
quantgrid worker [--worker-id]    one execution slot
quantgrid datasets                list / materialize recorded synthetic datasets
quantgrid verify BATCH_ID         prove one logical result per expected task
quantgrid gc                      delete stale temp artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _migrate() -> int:
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[2]
    ini = root / "alembic.ini"
    if not ini.exists():  # container layout: /app/alembic.ini
        ini = Path.cwd() / "alembic.ini"
    cfg = Config(str(ini))
    cfg.set_main_option("script_location", str(ini.parent / "migrations"))
    command.upgrade(cfg, "head")
    print("migrations applied")
    return 0


def _datasets(materialize: bool) -> int:
    from quantgrid.config import get_settings
    from quantgrid.domain.datasets import get_catalog
    from quantgrid.domain.datasets import materialize as mat

    for info in get_catalog().list():
        line = f"{info.name:24} {info.n_bars:>8} bars  sha256={info.sha256}"
        if materialize:
            line += f"  -> {mat(info.name, Path(get_settings().data_dir))}"
        print(line)
    return 0


def _verify(batch_id: str, as_json: bool, recompute: int) -> int:
    from quantgrid.config import get_settings
    from quantgrid.results.verify import verify_batch
    from quantgrid.storage.artifacts import LocalArtifactStore
    from quantgrid.storage.db import make_engine

    s = get_settings()
    report = verify_batch(
        make_engine(application_name="qg-verify"),
        LocalArtifactStore.from_settings(s),
        batch_id,
        recompute_sample=recompute,
    )
    print(json.dumps(report.to_dict(), indent=2, default=str) if as_json else report.summary())
    return 0 if report.ok else 1


def _gc(older_than_s: float) -> int:
    from quantgrid.storage.artifacts import LocalArtifactStore

    print(f"removed {LocalArtifactStore.from_settings().gc_temp(older_than_s)} stale temp file(s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quantgrid", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    api = sub.add_parser("api")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    sub.add_parser("coordinator")
    worker = sub.add_parser("worker")
    worker.add_argument("--worker-id")
    ds = sub.add_parser("datasets")
    ds.add_argument("--materialize", action="store_true")
    ver = sub.add_parser("verify")
    ver.add_argument("batch_id")
    ver.add_argument("--json", action="store_true")
    ver.add_argument("--recompute", type=int, default=0, help="re-run N tasks and compare bytes")
    gc = sub.add_parser("gc")
    gc.add_argument("--older-than-s", type=float, default=3600)
    args, rest = parser.parse_known_args(argv)

    if args.cmd == "migrate":
        return _migrate()
    if args.cmd == "api":
        import uvicorn

        uvicorn.run("quantgrid.api.app:create_app", factory=True, host=args.host, port=args.port)
        return 0
    if args.cmd == "coordinator":
        from quantgrid.coordinator.main import main as coordinator_main

        return coordinator_main(rest)
    if args.cmd == "worker":
        from quantgrid.workers.main import main as worker_main

        return worker_main((["--worker-id", args.worker_id] if args.worker_id else []) + rest)
    if args.cmd == "datasets":
        return _datasets(args.materialize)
    if args.cmd == "verify":
        return _verify(args.batch_id, args.json, args.recompute)
    return _gc(args.older_than_s)


if __name__ == "__main__":
    sys.exit(main())
