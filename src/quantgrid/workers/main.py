"""Worker process entry point: ``python -m quantgrid.workers.main [--worker-id ID]``.

One worker process = one execution slot (Celery ``solo`` pool). Scale out by running
more processes/containers; parallelism never comes from threads inside a worker, and
numeric libraries should run single-threaded (set ``OMP_NUM_THREADS=1`` etc. in the
environment *before* the process starts - the Dockerfile and devstack do this).
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantgrid worker", description=__doc__)
    parser.add_argument("--worker-id", default=None, help="stable worker id (default host-pid)")
    parser.add_argument("--loglevel", default=None, help="override QG_LOG_LEVEL")
    args = parser.parse_args(argv)

    if args.worker_id:
        os.environ["QG_WORKER_ID"] = args.worker_id
    from quantgrid.config import get_settings, reset_settings_cache

    reset_settings_cache()
    settings = get_settings()
    level = (args.loglevel or settings.log_level).upper()

    try:
        from quantgrid.observability.logging import configure_logging
        from quantgrid.observability.metrics import start_metrics_server
        from quantgrid.observability.tracing import init_tracing

        configure_logging(level, settings.log_format, service="quantgrid-worker")
        init_tracing("quantgrid-worker", settings.otel_exporter, settings.otel_endpoint)
        start_metrics_server(settings.worker_metrics_port)
    except ImportError:  # pragma: no cover - observability is part of the package
        pass

    from quantgrid.workers.celery_app import celery_app
    from quantgrid.workers.runtime import get_runtime

    runtime = get_runtime()  # fixes the worker id before Celery starts
    celery_app.worker_main(
        [
            "worker",
            "--pool=solo",
            "--concurrency=1",
            f"--hostname={runtime.worker_id}@%h",
            f"--queues={settings.queue_name}",
            "--without-gossip",
            "--without-mingle",
            f"--loglevel={level}",
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
