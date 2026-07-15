from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

from launcher import (  # noqa: E402
    Window,
    build_docker_command,
    build_local_command,
    load_config,
)


def test_local_and_docker_commands_share_workflow_parameters():
    config = load_config()
    values = {
        item["name"]: item.get("default") for item in config.get("parameters", [])
    }
    values.update({"model": "stardist:SD_Foci_Finn", "target": "foci"})
    local = build_local_command(
        config, values, "inputfolder", "outputfolder", "python-test"
    )
    docker = build_docker_command(
        config, values, "inputfolder", "outputfolder", gpu=True
    )

    assert local[:2] == ["python-test", str(Path("wrapper.py").resolve())]
    assert local[local.index("--infolder") + 1] == "inputfolder"
    assert local[local.index("--outfolder") + 1] == "outputfolder"
    assert local[local.index("--model") + 1] == "stardist:SD_Foci_Finn"
    assert "docker" not in local
    assert docker[:3] == ["docker", "run", "--rm"]
    assert "w_cisegmentation:latest" in docker
    assert "cellularimagingcf/w_cisegmentation:latest" not in docker
    assert docker[docker.index("--model") + 1] == "stardist:SD_Foci_Finn"


def test_launcher_exposes_both_run_buttons():
    app = QApplication.instance() or QApplication([])
    window = Window()
    labels = {button.text() for button in window.findChildren(QPushButton)}
    assert {"Run Docker", "Run Locally"} <= labels
    assert "Docker:" in window.preview.toPlainText()
    assert "Local Python:" in window.preview.toPlainText()
    window.close()
    app.processEvents()
