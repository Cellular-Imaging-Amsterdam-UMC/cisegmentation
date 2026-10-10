"""Pool cleanup must wait for OS exit, without competing with its reaper."""

import multiprocessing
from types import SimpleNamespace

import psutil
import pytest

from cisegmentation import process_pool


@pytest.mark.parametrize("exited_before_cleanup", [False, True])
def test_cleanup_uses_sentinel_and_never_joins_or_polls_worker(
    monkeypatch, exited_before_cleanup
):
    reader, writer = multiprocessing.Pipe(duplex=False)
    actions = []
    worker = SimpleNamespace(
        pid=777,
        sentinel=reader,
        join=lambda **kw: pytest.fail("Executor owns join"),
        is_alive=lambda: pytest.fail("Racy liveness poll"),
    )

    def terminate():
        actions.append("terminate")
        writer.close()

    process = SimpleNamespace(
        children=lambda **kw: [],
        terminate=terminate,
        kill=lambda: pytest.fail("Exited worker must not be killed"),
    )

    def lookup(pid):
        assert pid == worker.pid
        actions.append("lookup")
        return process

    monkeypatch.setattr(psutil, "Process", lookup)
    pool = SimpleNamespace(
        _processes={777: worker}, shutdown=lambda **kw: actions.append(("shutdown", kw))
    )
    try:
        if exited_before_cleanup:
            writer.close()
        process_pool.close_pool(pool, abort=True)
        assert actions == ([] if exited_before_cleanup else ["lookup", "terminate"]) + [
            ("shutdown", {"wait": True, "cancel_futures": True})
        ]
    finally:
        reader.close()
        writer.close()


def test_cleanup_kills_only_workers_with_unready_exit_handles(monkeypatch):
    actions = []
    worker = SimpleNamespace(pid=777)
    readings = iter([[worker], [worker], []])
    monkeypatch.setattr(
        process_pool, "_wait_for_workers", lambda *a, **kw: next(readings)
    )
    process = SimpleNamespace(
        children=lambda **kw: [],
        terminate=lambda: actions.append("terminate"),
        kill=lambda: actions.append("kill"),
    )
    monkeypatch.setattr(psutil, "Process", lambda _: process)
    pool = SimpleNamespace(
        _processes={777: worker}, shutdown=lambda **kw: actions.append("shutdown")
    )
    process_pool.close_pool(pool, abort=True)
    assert actions == ["terminate", "kill", "shutdown"]


def test_cleanup_refuses_retry_if_exit_handle_remains_unready(monkeypatch):
    worker = SimpleNamespace(pid=777)
    monkeypatch.setattr(process_pool, "_wait_for_workers", lambda *a, **kw: [worker])
    process = SimpleNamespace(
        children=lambda **kw: [], terminate=lambda: None, kill=lambda: None
    )
    monkeypatch.setattr(psutil, "Process", lambda _: process)
    pool = SimpleNamespace(
        _processes={777: worker},
        shutdown=lambda **kw: pytest.fail("Live workers must block retry"),
    )
    with pytest.raises(RuntimeError, match="still alive; refusing to retry"):
        process_pool.close_pool(pool, abort=True)
