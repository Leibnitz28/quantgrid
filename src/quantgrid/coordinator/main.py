"""Coordinator entry point: ``python -m quantgrid.coordinator.main``.

Graceful shutdown on SIGTERM/SIGINT: finish the current tick, release the leader lock,
exit 0. Safe to run several replicas (one leader, the rest standby).
"""

from __future__ import annotations

import argparse
import signal
import sys
from types import FrameType


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantgrid coordinator", description=__doc__)
    parser.parse_args(argv)

    from quantgrid.config import get_settings
    from quantgrid.coordinator.loop import Coordinator, celery_sender
    from quantgrid.storage.db import make_engine

    settings = get_settings()
    try:
        from quantgrid.observability.logging import configure_logging
        from quantgrid.observability.metrics import start_metrics_server
        from quantgrid.observability.tracing import init_tracing

        configure_logging(settings.log_level, settings.log_format, service="quantgrid-coordinator")
        init_tracing("quantgrid-coordinator", settings.otel_exporter, settings.otel_endpoint)
        start_metrics_server(settings.coordinator_metrics_port)
    except ImportError:  # pragma: no cover
        pass

    engine = make_engine(pool_size=3, application_name="qg-coordinator")
    coordinator = Coordinator(settings, engine, celery_sender(settings.queue_name))

    def _stop(signum: int, _frame: FrameType | None) -> None:
        coordinator.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        coordinator.run()
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
