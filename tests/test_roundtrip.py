from __future__ import annotations

import json
from pathlib import Path

import pytest

from cisegmentation.roundtrip import (
    descriptor_url,
    first_ome_zarr,
    image_manifest_matches,
    is_hcs_store,
    redact,
    roundtrip_command,
    update_biomero_config,
)


def test_first_ome_zarr_is_top_level_and_deterministic(tmp_path: Path):
    (tmp_path / "b.ome.zarr").mkdir()
    (tmp_path / "A.ome.zarr").mkdir()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "0.ome.zarr").mkdir()
    assert first_ome_zarr(tmp_path).name == "A.ome.zarr"


def test_first_ome_zarr_requires_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="top-level"):
        first_ome_zarr(tmp_path)


def test_hcs_detection(tmp_path: Path):
    store = tmp_path / "plate.ome.zarr"
    store.mkdir()
    (store / ".zattrs").write_text(json.dumps({"plate": {"wells": []}}))
    assert is_hcs_store(store)
    (store / ".zattrs").write_text(json.dumps({"multiscales": []}))
    assert not is_hcs_store(store)


def test_config_registration_is_idempotent():
    original = """[WORKFLOWS]\nfoo = foo\n\n[UI]\nzarr_workflows = [\"foo\"]\nplate_workflows = []\n"""
    url = descriptor_url("abc123")
    updated, changed = update_biomero_config(original, url)
    assert changed
    assert "cisegmentation_repo = " + url in updated
    assert 'zarr_workflows = ["foo","cisegmentation"]' in updated
    assert 'plate_workflows = ["cisegmentation"]' in updated
    again, changed_again = update_biomero_config(updated, url)
    assert not changed_again
    assert again == updated


def test_manifest_requires_all_identity_fields():
    current = {
        "manifest_version": 2,
        "docker_image_id": "sha256:1",
        "descriptor_sha256": "a",
        "git_commit": "b",
        "image_tag": "latest",
    }
    assert image_manifest_matches(current, dict(current))
    remote = dict(current)
    remote["git_commit"] = "other"
    assert not image_manifest_matches(current, remote)
    assert not image_manifest_matches(current, None)


def test_redaction_and_roundtrip_serialization():
    assert redact("password=secret", ["secret"]) == "password=<redacted>"
    assert redact("BIOMERO OMERO_PASSWORD=omero", ["omero"]) == "BIOMERO OMERO_PASSWORD=<redacted>"
    command = roundtrip_command("python", "runner.py", "in", "out", {"model": "x", "benchmark": True}, False)
    assert command[-2:] == ["--gpu", "false"]
    payload = json.loads(command[command.index("--parameters-json") + 1])
    assert payload == {"model": "x", "benchmark": True}
