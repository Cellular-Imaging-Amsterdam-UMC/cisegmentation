from __future__ import annotations

from pathlib import Path


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
