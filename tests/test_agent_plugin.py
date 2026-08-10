from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
SKILLS = ROOT / "skills"
NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, frontmatter, body = text.split("---", 2)
    assert body.strip()
    return yaml.safe_load(frontmatter)


def test_agent_plugin_manifest_and_skills():
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["$schema"] == (
        "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    )
    assert manifest["version"] == "0.5.0"
    declared = manifest["extensions"]["nl.bioimaging.biomero"]["skills"]
    assert set(declared) == {path.name for path in SKILLS.iterdir() if path.is_dir()}

    allowed_frontmatter = {
        "name", "description", "license", "compatibility", "metadata", "allowed-tools"
    }
    allowed_suffixes = {".md", ".txt", ".json", ".yaml", ".yml"}
    for name, consumer_metadata in declared.items():
        assert NAME_PATTERN.fullmatch(name)
        skill_dir = SKILLS / name
        frontmatter = _frontmatter(skill_dir / "SKILL.md")
        assert set(frontmatter) <= allowed_frontmatter
        assert frontmatter["name"] == name
        assert 1 <= len(str(frontmatter["description"])) <= 1024
        assert all(isinstance(value, str) for value in frontmatter["metadata"].values())
        for resource in consumer_metadata.get("required_resources", []):
            assert (skill_dir / resource).is_file()
        for path in skill_dir.rglob("*"):
            if path.is_file() and path.name != "SKILL.md":
                assert path.parent == skill_dir / "references"
                assert path.suffix.lower() in allowed_suffixes


def test_measurement_matching_is_specific_and_remote_capable():
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    measurement = manifest["extensions"]["nl.bioimaging.biomero"]["skills"][
        "analyze-cisegmentation-measurements"
    ]
    assert measurement["match"]["extensions"] == [".duckdb", ".sqlite", ".sqlite3"]
    assert ".csv" not in measurement["match"]["extensions"]
    assert measurement["preferred_capabilities"] == ["omero-data-query-v1"]
    assert "required_capabilities" not in measurement


def test_workflow_skill_documents_every_descriptor_parameter():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    documented = (
        SKILLS / "use-cisegmentation-workflow" / "references" / "PARAMETERS.md"
    ).read_text(encoding="utf-8")
    for parameter in config["parameters"]:
        assert f"`{parameter['name']}`" in documented


def test_versions_are_synchronized():
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert manifest["version"] == "0.5.0"
    assert 'version = "0.5.0"' in pyproject
    assert (ROOT / "version.txt").read_text(encoding="utf-8").strip() == "v0.5.0"
    assert config["docker_image"]["tag"] == "v0.5.0"


def test_skill_content_is_platform_neutral():
    forbidden = ("omero", "biomero", "jupyterlite", "browser-local")
    for path in SKILLS.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8").lower()
            assert not any(term in text for term in forbidden), path
