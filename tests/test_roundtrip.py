from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from cisegmentation.roundtrip import (
    RoundtripRunner,
    descriptor_url,
    docker_build_state_matches,
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


def test_local_docker_state_requires_source_and_image_identity():
    state = {"source_id": "source-a", "image_id": "image-a"}
    assert docker_build_state_matches(state, "source-a", "image-a")
    assert not docker_build_state_matches(state, "source-b", "image-a")
    assert not docker_build_state_matches(state, "source-a", "image-b")
    assert not docker_build_state_matches(None, "source-a", "image-a")


def test_command_stream_replaces_invalid_output_bytes_without_hanging(tmp_path):
    runner = RoundtripRunner(
        Path.cwd(),
        tmp_path,
        tmp_path,
        {},
        biomero_root=tmp_path,
        emit=lambda _line: None,
    )
    result = runner.run_cmd(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'before\\x90after\\n')",
        ],
        "invalid-output.log",
        timeout=5,
    )
    assert result.stdout == "before�after\n"


def test_live_output_falls_back_when_console_cannot_encode_unicode(tmp_path):
    emitted: list[str] = []

    def ascii_console(line: str) -> None:
        line.encode("ascii")
        emitted.append(line)

    runner = RoundtripRunner(
        Path.cwd(), tmp_path, tmp_path, {}, biomero_root=tmp_path, emit=ascii_console
    )
    runner._emit("OMERO result ━ complete")
    assert emitted == ["OMERO result ? complete"]


def test_command_output_can_be_logged_without_echoing_to_ui(tmp_path):
    emitted: list[str] = []
    runner = RoundtripRunner(
        Path.cwd(), tmp_path, tmp_path, {}, biomero_root=tmp_path, emit=emitted.append
    )
    result = runner.run_cmd(
        [sys.executable, "-c", "print('diagnostic output')"],
        "diagnostic.log",
        timeout=5,
        echo_output=False,
    )
    assert result.stdout == "diagnostic output\n"
    assert emitted == []
    assert "diagnostic output" in (runner.log_dir / "diagnostic.log").read_text(
        encoding="utf-8"
    )


def test_redaction_and_roundtrip_serialization():
    assert redact("password=secret", ["secret"]) == "password=<redacted>"
    assert redact("BIOMERO OMERO_PASSWORD=omero", ["omero"]) == "BIOMERO OMERO_PASSWORD=<redacted>"
    command = roundtrip_command("python", "runner.py", "in", "out", {"model": "x", "benchmark": True}, False)
    assert command[-2:] == ["--gpu", "false"]
    payload = json.loads(command[command.index("--parameters-json") + 1])
    assert payload == {"model": "x", "benchmark": True}


def test_roundtrip_applies_black_background_glasbey_to_result_images(
    tmp_path, monkeypatch
):
    runner = RoundtripRunner(
        Path.cwd(), tmp_path, tmp_path, {}, biomero_root=tmp_path, emit=lambda _: None
    )
    captured = {}

    def fake_python(code, environment, log_name):
        captured.update(code=code, environment=environment, log_name=log_name)
        return environment["IMAGE_IDS"]

    monkeypatch.setattr(runner, "_container_python", fake_python)
    runner._apply_label_lut("Dataset", [41, 42])

    assert "glasbey_inverted.lut" in captured["code"]
    assert captured["environment"]["IMAGE_IDS"] == "41,42"
    assert runner.summary["result_lookup_table"] == "glasbey_inverted.lut"
