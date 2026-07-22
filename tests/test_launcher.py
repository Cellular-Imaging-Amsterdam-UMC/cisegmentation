from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QWheelEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

import launcher  # noqa: E402

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
    tag = config["docker_image"]["tag"]
    assert f"w_cisegmentation:{tag}" in docker
    assert f"cellularimagingcf/w_cisegmentation:{tag}" not in docker
    assert docker[docker.index("--foci-model-1") + 1] == "stardist:SD_Foci_Finn"
    assert "--multi-step" not in local


def test_launcher_serializes_skip_and_fourth_foci_slot():
    config = load_config()
    values = {
        item["name"]: item.get("default") for item in config.get("parameters", [])
    }
    values.update(
        {"cell_model": "skip", "foci_model_4": "spotiflow:general", "foci_channel_4": 3}
    )
    command = build_local_command(
        config, values, "inputfolder", "outputfolder", "python-test"
    )
    assert command[command.index("--cell-model") + 1] == "skip"
    assert command[command.index("--foci-model-4") + 1] == "spotiflow:general"
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


def test_launcher_defaults_to_only_step1_and_never_grays_controls():
    app = QApplication.instance() or QApplication([])
    window = Window()
    assert window.widgets["cell_model"].currentData() == "cellpose3:cyto3"
    assert window.widgets["nucleus_model"].currentData() == "skip"
    assert all(window.widgets[f"foci_model_{slot}"].currentData() == "skip" for slot in range(1, 5))
    for name in (
        "cell_channel",
        "cell_nuclei_channel",
        "cell_expansion_distance",
        "nucleus_channel",
        "foci_channel_1",
    ):
        assert window.widgets[name].isEnabled()
    window.widgets["cell_model"].setCurrentIndex(
        window.widgets["cell_model"].findData("skip")
    )
    assert all(widget.isEnabled() for widget in window.widgets.values())
    window.close()
    app.processEvents()


def test_collapsing_advanced_parameters_restores_compact_window_height():
    app = QApplication.instance() or QApplication([])
    window = Window()
    window.show()
    app.processEvents()
    window.adjustSize()
    app.processEvents()
    collapsed_height = window.height()

    window.advanced_panel.toggle.click()
    app.processEvents()
    app.processEvents()
    expanded_height = window.height()
    assert expanded_height > collapsed_height

    window.advanced_panel.toggle.click()
    app.processEvents()
    app.processEvents()
    assert window.height() == collapsed_height

    window.close()
    app.processEvents()


def test_mouse_wheel_never_changes_parameter_values():
    app = QApplication.instance() or QApplication([])
    window = Window()

    def wheel_up(widget):
        event = QWheelEvent(
            QPointF(1, 1),
            QPointF(1, 1),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        QApplication.sendEvent(widget, event)

    combo = window.widgets["cell_model"]
    combo_before = combo.currentIndex()
    wheel_up(combo)
    assert combo.currentIndex() == combo_before

    integer = window.widgets["cell_channel"]
    integer_before = integer.value()
    wheel_up(integer)
    assert integer.value() == integer_before

    floating = window.widgets["cellprob_threshold"]
    floating_before = floating.value()
    wheel_up(floating)
    assert floating.value() == floating_before

    window.close()
    app.processEvents()


def test_launcher_restore_migrates_legacy_settings(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "values": {
                    "cell_step": False,
                    "cell_model": "cellpose3:cyto3",
                    "nucleus_step": False,
                    "nucleus_model": "cellpose3:nuclei",
                    "foci_step_1": True,
                    "foci_model_1": "stardist:SD_Foci_Finn",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "SETTINGS", settings_file)
    app = QApplication.instance() or QApplication([])
    window = Window()
    window.restore()
    assert window.widgets["cell_model"].currentData() == "skip"
    assert window.widgets["nucleus_model"].currentData() == "skip"
    assert window.widgets["foci_model_1"].currentData() == "stardist:SD_Foci_Finn"
    window.close()
    app.processEvents()
