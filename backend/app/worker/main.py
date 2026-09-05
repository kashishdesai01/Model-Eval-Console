import logging
import os
import signal
import socket
import threading
import time
from uuid import uuid4

from prometheus_client import start_http_server

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.metrics import RUN_DURATION, RUNS
from app.db.session import session_factory
from app.inference.service import InferenceService
from app.worker.queue import Claim, LostLease, claim_run, fail, heartbeat
from app.worker.runner import evaluate

logger = logging.getLogger(__name__)


def maintain_lease(claim: Claim, done: threading.Event) -> None:
    settings = get_settings()
    while not done.wait(settings.lease_seconds / 3):
        try:
            with session_factory().begin() as session:
                heartbeat(session, claim, settings.lease_seconds)
        except Exception:
            logger.exception("Lease heartbeat failed", extra={"run_id": claim.run_id})
            return


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda _signal, _frame: stop.set())
    start_http_server(int(os.environ.get("MEC_WORKER_METRICS_PORT", "8001")))
    worker_id = f"{socket.gethostname()}:{uuid4()}"
    inference = InferenceService()
    while not stop.is_set():
        with session_factory().begin() as session:
            claim = claim_run(session, worker_id, settings.lease_seconds)
        if claim is None:
            stop.wait(1)
            continue
        done = threading.Event()
        lease_thread = threading.Thread(target=maintain_lease, args=(claim, done), daemon=True)
        lease_thread.start()
        started = time.monotonic()
        try:
            evaluate(claim, session_factory(), inference)
            RUNS.labels("succeeded").inc()
        except LostLease:
            logger.warning("Stopped writing after claim loss", extra={"run_id": claim.run_id})
        except Exception as exc:
            logger.exception(
                "Evaluation attempt failed",
                extra={"run_id": claim.run_id, "attempt": claim.attempt},
            )
            try:
                with session_factory().begin() as session:
                    fail(session, claim, str(exc))
            except LostLease:
                pass
            RUNS.labels("failed").inc()
        finally:
            done.set()
            lease_thread.join(timeout=5)
            RUN_DURATION.observe(time.monotonic() - started)


if __name__ == "__main__":
    main()
