from __future__ import annotations

import hashlib

from tools.docker_runtime_cache_id import runtime_cache_id


def test_runtime_cache_id_is_stable_and_includes_model_identity():
    first = runtime_cache_id("models-a")
    assert first == runtime_cache_id("models-a")
    assert first != runtime_cache_id("models-b")
    assert len(first) == hashlib.sha256().digest_size * 2
