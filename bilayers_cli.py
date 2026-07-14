from __future__ import annotations

import argparse
from pathlib import Path
import shlex
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Bilayers configuration must be a mapping")
    return config


def validate_config(config: dict[str, Any]) -> list[str]:
    errors = []
    for key in (
        "citations",
        "docker_image",
        "algorithm_folder_name",
        "exec_function",
        "inputs",
        "outputs",
        "parameters",
        "display_only",
    ):
        if key not in config:
            errors.append(f"Missing top-level key: {key}")
    for section in ("inputs", "outputs", "parameters"):
        for index, item in enumerate(config.get(section, []) or []):
            if not isinstance(item, dict) or not item.get("name"):
                errors.append(f"{section}[{index}] must have a name")
            if section != "outputs" and not item.get("cli_tag"):
                errors.append(f"{section}[{index}] must have a cli_tag")
    return errors


def generate_cli_command(
    config: dict[str, Any], values: dict[str, Any] | None = None
) -> str:
    values = values or {}
    command = shlex.split(str(config["exec_function"]["cli_command"]))
    items = []
    for section in ("inputs", "outputs", "parameters"):
        items.extend(config.get(section, []) or [])
    items.extend(config.get("exec_function", {}).get("hidden_args", []) or [])
    for item in sorted(items, key=lambda value: int(value.get("cli_order", 0))):
        tag = item.get("cli_tag")
        value = values.get(item.get("name"), item.get("value", item.get("default")))
        if not tag or value in (None, "", False):
            continue
        append = item.get("append_value", item.get("type") != "checkbox")
        command.append(str(tag))
        if append:
            if isinstance(value, list):
                value = ",".join(str(entry) for entry in value)
            command.append(str(value))
    return shlex.join(command)


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("show", "validate", "generate"):
        child = sub.add_parser(name)
        child.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        errors = validate_config(config)
        for error in errors:
            print(error)
        return int(bool(errors))
    if args.command == "generate":
        print(generate_cli_command(config))
    else:
        print(yaml.safe_dump(config, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
