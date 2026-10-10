"""Release only workers owned by a failed pool before resource retries."""

from __future__ import annotations


def close_pool(pool, *, abort=False):
    if abort:
        # Python 3.11/3.12 lack terminate_workers(). Keep the compatibility
        # access isolated here and capture ownership before shutdown clears it.
        workers = list((getattr(pool, "_processes", None) or {}).values())
        import psutil

        descendants = []
        for worker in workers:
            if not worker.is_alive():
                continue
            try:
                descendants.extend(psutil.Process(worker.pid).children(recursive=True))
            except psutil.Error:
                pass
        for process in descendants:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
            if worker.is_alive():
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
    pool.shutdown(wait=True, cancel_futures=True)
