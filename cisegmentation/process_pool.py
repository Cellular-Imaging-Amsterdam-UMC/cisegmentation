"""Release only workers owned by a failed pool before resource retries."""

from __future__ import annotations

import time
from multiprocessing.connection import wait


def _wait_for_workers(workers, timeout):
    """Wait for OS exit handles without reaping the executor's children."""
    pending = {worker.sentinel: worker for worker in workers}
    deadline = time.monotonic() + timeout
    while pending:
        ready = wait(list(pending), timeout=max(0, deadline - time.monotonic()))
        for sentinel in ready:
            pending.pop(sentinel)
        if not ready:
            break
    return list(pending.values())


def close_pool(pool, *, abort=False):
    if abort:
        # Python 3.11/3.12 lack terminate_workers(). Keep the compatibility
        # access isolated here and capture ownership before shutdown clears it.
        workers = list((getattr(pool, "_processes", None) or {}).values())
        import psutil

        descendants, processes = [], {}
        for worker in _wait_for_workers(workers, timeout=0):
            try:
                # psutil signals also check process identity, protecting against
                # PID reuse if the executor has already reaped this child.
                process = psutil.Process(worker.pid)
                processes[worker.pid] = process
                descendants.extend(process.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
        for process in descendants:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        for process in processes.values():
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        remaining = _wait_for_workers(workers, timeout=5)
        for worker in remaining:
            process = processes.get(worker.pid)
            if process is not None:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
        if _wait_for_workers(remaining, timeout=5):
            raise RuntimeError(
                "Failed inference worker is still alive; refusing to retry"
            )
        _, alive = psutil.wait_procs(descendants, timeout=5)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(alive, timeout=5)
        if alive:
            raise RuntimeError("Failed tile workers are still alive; refusing to retry")
    # The executor manager thread remains the only owner of Process.join()/
    # waitpid(). Competing joins can report stale liveness on Linux even after
    # SIGKILL. Sentinels confirm OS exit first; shutdown then joins the manager.
    pool.shutdown(wait=True, cancel_futures=True)
