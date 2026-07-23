from __future__ import annotations

from pathlib import Path

import pytest

from tools.sync_config_version import sync_config_version


ROOT = Path(__file__).resolve().parents[1]


def test_publication_scripts_are_tracked_safe_tools():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "pushdocker.cmd" not in ignored
    assert "release_github.cmd" not in ignored

    docker = (ROOT / "pushdocker.cmd").read_text(encoding="utf-8")
    assert 'if "%DRY_RUN%"=="0" if "%CONFIRMED%"=="0"' in docker
    assert "--dry-run" in docker
    assert "--yes" in docker
    assert "cellularimagingcf/w_cisegmentation" in docker
    assert "Version must be SemVer with an optional v prefix" in docker
    assert "\\Av?(?:0|[1-9]\\d*)" in docker

    github = (ROOT / "release_github.cmd").read_text(encoding="utf-8")
    assert 'if "%DRY_RUN%"=="0" if "%CONFIRMED%"=="0"' in github
    assert "--dry-run" in github
    assert "--yes" in github
    assert "Cellular-Imaging-Amsterdam-UMC/cisegmentation.git" in github
    assert "version.txt must contain SemVer with an optional v prefix" in github
    assert "\\Av?(?:0|[1-9]\\d*)" in github
    assert "tools\\sync_config_version.py" in github
    assert 'git commit -m "Set config Docker tag to %TAG%"' in github
    assert "git push origin HEAD" in github


def test_release_version_sync_updates_only_docker_tag(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "docker_image:\n"
        "  org: example\n"
        "  name: image\n"
        "  tag: v1.2.2\n"
        "parameters:\n"
        "- {name: tag, default: untouched}\n",
        encoding="utf-8",
    )

    assert sync_config_version(config, "v1.2.3", check=True) is True
    assert sync_config_version(config, "v1.2.3") is True
    assert sync_config_version(config, "v1.2.3", check=True) is False
    assert config.read_text(encoding="utf-8") == (
        "docker_image:\n"
        "  org: example\n"
        "  name: image\n"
        "  tag: v1.2.3\n"
        "parameters:\n"
        "- {name: tag, default: untouched}\n"
    )


def test_release_version_sync_rejects_invalid_version(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("docker_image:\n  tag: v1.2.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid release version"):
        sync_config_version(config, "latest")
