from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
SETTINGS = ROOT / ".last_launcher_settings.json"


class MultiSelectList(QListWidget):
    def __init__(self, options: list[dict], defaults: object):
        super().__init__()
        self.setMaximumHeight(125)
        selected = set(defaults if isinstance(defaults, list) else [defaults])
        for option in options:
            item = QListWidgetItem(str(option.get("label", option.get("value"))))
            item.setData(Qt.ItemDataRole.UserRole, option.get("value"))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if option.get("value") in selected
                else Qt.CheckState.Unchecked
            )
            self.addItem(item)

    def values(self) -> list[str]:
        return [
            str(self.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.count())
            if self.item(i).checkState() == Qt.CheckState.Checked
        ]


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def build_docker_command(
    config: dict, values: dict, input_dir: str, output_dir: str, gpu: bool = True
) -> list[str]:
    image = config["docker_image"]
    image_name = f"{image['org']}/{image['name']}:{image.get('tag', 'latest')}"
    command = ["docker", "run", "--rm"]
    if gpu:
        command += ["--gpus", "all"]
    command += [
        "-v",
        f"{input_dir}:/data/in",
        "-v",
        f"{output_dir}:/data/out",
        image_name,
    ]
    for item in sorted(
        config.get("parameters", []), key=lambda entry: int(entry.get("cli_order", 0))
    ):
        value = values.get(item["name"], item.get("default"))
        if value in (None, "", False, []):
            continue
        command.append(str(item["cli_tag"]))
        if item.get("type") != "checkbox" or item.get("append_value", False):
            if isinstance(value, list):
                value = ",".join(value)
            command.append(str(value))
    command.append("--local")
    return command


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.widgets: dict[str, QWidget] = {}
        self.setWindowTitle("CI Segmentation Docker Launcher")
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        folders = QFormLayout()
        layout.addLayout(folders)
        self.input_path = QLineEdit(str(ROOT / "tests" / "data"))
        self.output_path = QLineEdit(str(ROOT / "outputs"))
        folders.addRow("Input folder", self._folder_row(self.input_path))
        folders.addRow("Output folder", self._folder_row(self.output_path))
        form = QFormLayout()
        layout.addLayout(form)
        for parameter in self.config.get("parameters", []):
            widget = self._widget(parameter)
            self.widgets[parameter["name"]] = widget
            form.addRow(parameter.get("label", parameter["name"]), widget)
        self.gpu = QCheckBox("Use NVIDIA GPU")
        self.gpu.setChecked(True)
        layout.addWidget(self.gpu)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(110)
        layout.addWidget(self.preview)
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        run = QPushButton("Run")
        run.clicked.connect(self.run)
        buttons.addWidget(run)
        save = QPushButton("Save settings")
        save.clicked.connect(self.save)
        buttons.addWidget(save)
        restore = QPushButton("Restore settings")
        restore.clicked.connect(self.restore)
        buttons.addWidget(restore)
        self.refresh()
        for widget in self.widgets.values():
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.refresh)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.refresh)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.refresh)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self.refresh)
            elif isinstance(widget, MultiSelectList):
                widget.itemChanged.connect(self.refresh)

    def _folder_row(self, edit: QLineEdit) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        button = QPushButton("Browse")
        button.clicked.connect(lambda: self._browse(edit))
        row.addWidget(button)
        return box

    def _browse(self, edit: QLineEdit):
        selected = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if selected:
            edit.setText(selected)
            self.refresh()

    def _widget(self, spec: dict) -> QWidget:
        if spec.get("multiselect"):
            return MultiSelectList(spec.get("options", []), spec.get("default", []))
        if spec.get("type") == "checkbox":
            widget = QCheckBox()
            widget.setChecked(bool(spec.get("default")))
            return widget
        if spec.get("options"):
            widget = QComboBox()
            for option in spec["options"]:
                widget.addItem(str(option["label"]), option["value"])
            index = widget.findData(spec.get("default"))
            widget.setCurrentIndex(max(index, 0))
            return widget
        if spec.get("type") == "integer":
            widget = QSpinBox()
            widget.setRange(
                int(spec.get("minimum", -999999)), int(spec.get("maximum", 999999))
            )
            widget.setValue(int(spec.get("default", 0)))
            return widget
        if spec.get("type") == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(6)
            widget.setRange(
                float(spec.get("minimum", -999999)), float(spec.get("maximum", 999999))
            )
            widget.setValue(float(spec.get("default", 0)))
            return widget
        widget = QLineEdit(str(spec.get("default", "")))
        return widget

    def values(self) -> dict:
        values = {}
        for name, widget in self.widgets.items():
            if isinstance(widget, MultiSelectList):
                values[name] = widget.values()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                values[name] = widget.value()
            else:
                values[name] = widget.text()
        return values

    def command(self) -> list[str]:
        return build_docker_command(
            self.config,
            self.values(),
            self.input_path.text(),
            self.output_path.text(),
            self.gpu.isChecked(),
        )

    def refresh(self):
        self.preview.setPlainText(subprocess.list2cmdline(self.command()))

    def run(self):
        self.save()
        subprocess.Popen(self.command(), cwd=ROOT)

    def save(self):
        SETTINGS.write_text(
            json.dumps(
                {
                    "values": self.values(),
                    "input": self.input_path.text(),
                    "output": self.output_path.text(),
                    "gpu": self.gpu.isChecked(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def restore(self):
        if not SETTINGS.exists():
            return
        data = json.loads(SETTINGS.read_text(encoding="utf-8"))
        self.input_path.setText(data.get("input", ""))
        self.output_path.setText(data.get("output", ""))
        self.gpu.setChecked(data.get("gpu", True))
        for name, value in data.get("values", {}).items():
            widget = self.widgets.get(name)
            if isinstance(widget, QComboBox):
                widget.setCurrentIndex(max(0, widget.findData(value)))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, MultiSelectList):
                selected = set(value)
                for i in range(widget.count()):
                    widget.item(i).setCheckState(
                        Qt.CheckState.Checked
                        if widget.item(i).data(Qt.ItemDataRole.UserRole) in selected
                        else Qt.CheckState.Unchecked
                    )
        self.refresh()


def main() -> int:
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
