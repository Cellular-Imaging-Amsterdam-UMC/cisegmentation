from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
SKILLS = ROOT / "_agents" / "skills"


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, frontmatter, _body = text.split("---", 2)
    return yaml.safe_load(frontmatter)


def test_portable_skills_follow_biomero_catalog_contract():
    expected = {
        "analyze-cisegmentation-measurements": {
            "biomero-purpose": "attachment-analysis",
            "biomero-consumers": "omero-analysis-chat,omero-jupyterlite",
            "biomero-auto-activate": "true",
        },
        "use-cisegmentation-workflow": {
            "biomero-purpose": "workflow-operation",
            "biomero-consumers": "omero-biomero",
            "biomero-auto-activate": "false",
        },
    }
    allowed_suffixes = {".md", ".txt", ".json", ".yaml", ".yml"}

    assert {path.name for path in SKILLS.iterdir() if path.is_dir()} == set(expected)
    for name, required_metadata in expected.items():
        skill_dir = SKILLS / name
        metadata = _frontmatter(skill_dir / "SKILL.md")
        assert metadata["name"] == name
        assert metadata["metadata"]["version"].isdigit()
        assert all(
            isinstance(value, str) for value in metadata["metadata"].values()
        )
        for key, value in required_metadata.items():
            assert metadata["metadata"][key] == value

        for path in skill_dir.rglob("*"):
            if path.is_file() and path.name != "SKILL.md":
                assert path.parent == skill_dir / "references"
                assert path.suffix.lower() in allowed_suffixes


def test_workflow_skill_documents_every_descriptor_parameter():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    documented = (
        SKILLS
        / "use-cisegmentation-workflow"
        / "references"
        / "PARAMETERS.md"
    ).read_text(encoding="utf-8")

    for parameter in config["parameters"]:
        assert f"`{parameter['name']}`" in documented
