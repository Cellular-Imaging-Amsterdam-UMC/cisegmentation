from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_entry_points_pin_cisegmentation_environment():
    settings = json.loads((ROOT / ".vscode" / "settings.json").read_text(encoding="utf-8"))
    assert settings["python.defaultInterpreterPath"].endswith(
        "\\miniconda3\\envs\\cisegmentation\\python.exe"
    )
    assert settings["python-envs.terminal.autoActivationType"] == "command"
    assert settings["python.terminal.activateEnvironment"] is True
    assert settings["python-envs.defaultEnvManager"] == "ms-python.python:conda"

    launcher = (ROOT / "launch.cmd").read_text(encoding="utf-8")
    assert "%LOCALAPPDATA%\\miniconda3\\envs\\cisegmentation\\python.exe" in launcher
    assert '"%PYTHON_EXE%" "%~dp0launcher.py" %*' in launcher
    assert 'set "PYTHON_EXE=python"' not in launcher

    test_runner = (ROOT / "test.cmd").read_text(encoding="utf-8")
    assert "%LOCALAPPDATA%\\miniconda3\\envs\\cisegmentation\\python.exe" in test_runner
    assert '"%PYTHON_EXE%" -m pytest %*' in test_runner
    assert 'set "PYTHON_EXE=python"' not in test_runner
