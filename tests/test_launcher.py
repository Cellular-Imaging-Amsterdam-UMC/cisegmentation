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
    values.update(
        {
            "foci_step_1": True,
            "foci_model_1": "stardist:SD_Foci_Finn",
            "foci_channel_1": 2,
        }
    )
    local = build_local_command(
        config, values, "inputfolder", "outputfolder", "python-test"
    )
    docker = build_docker_command(
        config, values, "inputfolder", "outputfolder", gpu=True
    )

    assert local[:2] == ["python-test", str(Path("wrapper.py").resolve())]
    assert local[local.index("--infolder") + 1] == "inputfolder"
    assert local[local.index("--outfolder") + 1] == "outputfolder"
    assert local[local.index("--foci-model-1") + 1] == "stardist:SD_Foci_Finn"
    assert "docker" not in local
    assert docker[:3] == ["docker", "run", "--rm"]
    assert "w_cisegmentation:latest" in docker
    assert "cellularimagingcf/w_cisegmentation:latest" not in docker
    assert docker[docker.index("--foci-model-1") + 1] == "stardist:SD_Foci_Finn"
    assert "--multi-step" not in local


def test_launcher_serializes_disabled_steps_and_four_foci_slots():
    config = load_config()
    values = {
        item["name"]: item.get("default") for item in config.get("parameters", [])
    }
    values.update({"cell_step": False, "foci_step_4": True, "foci_channel_4": 3})
    command = build_local_command(
        config, values, "inputfolder", "outputfolder", "python-test"
    )
    assert command[command.index("--cell-step") + 1] == "False"
    assert command[command.index("--foci-step-4") + 1] == "True"
    assert command[command.index("--foci-channel-4") + 1] == "3"


def test_launcher_exposes_both_run_buttons():
    app = QApplication.instance() or QApplication([])
    window = Window()
    labels = {button.text() for button in window.findChildren(QPushButton)}
    assert {"Run Docker", "Run Locally"} <= labels
    assert "Docker:" in window.preview.toPlainText()
    assert "Local Python:" in window.preview.toPlainText()
    window.close()
    app.processEvents()


def test_launcher_defaults_to_only_step1_and_rejects_no_steps(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = Window()
    assert window.widgets["cell_step"].isChecked()
    assert not window.widgets["nucleus_step"].isChecked()
    assert all(
        not window.widgets[f"foci_step_{slot}"].isChecked()
        for slot in range(1, 5)
    )
    assert window._validate_run_selection() is True
    window.widgets["cell_step"].setChecked(False)
    messages = []
    monkeypatch.setattr(
        "launcher.QMessageBox.warning",
        lambda *args: messages.append(args[-1]),
    )
    assert window._validate_run_selection() is False
    assert "Select Cell Detection" in messages[0]
    window.close()
    app.processEvents()
