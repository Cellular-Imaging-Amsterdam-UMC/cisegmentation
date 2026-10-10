"""Keep default Docker builds/publication workflow-only and extras manual."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="Windows batch scripts")


def test_default_ci_builds_only_the_workflow_image():
    workflow = yaml.safe_load((ROOT / ".github/workflows/agent-plugin.yml").read_text())
    build = workflow["jobs"]["docker-build"]
    commands = "\n".join(step.get("run", "") for step in build["steps"])
    assert "strategy" not in build
    assert commands.count("docker build") == 1
    assert "--file Dockerfile" in commands
    assert "gradio" not in commands and "jupyter" not in commands


def test_optional_dockerfiles_have_self_contained_build_contexts():
    extras = ROOT / "bilayers_extra"
    for variant in ("gradio", "jupyter"):
        for name in (
            f"Dockerfile.{variant}",
            f"builddocker_{variant}.cmd",
            f"requirements_{variant}.txt",
        ):
            assert (extras / name).is_file()
            assert not (ROOT / name).exists()
        dockerfile = (extras / f"Dockerfile.{variant}").read_text()
        for line in dockerfile.splitlines():
            if line.startswith("COPY "):
                assert (extras / line.split()[1]).is_file()
    assert "bilayers_extra/" in (ROOT / ".dockerignore").read_text().splitlines()


def command_repo(tmp_path):
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    (repo / "version.txt").write_text("v9.8.7\n")
    (repo / "config.yaml").write_text(
        "docker_image:\n  org: cellularimagingcf\n  name: w_cisegmentation\n"
    )
    extras = repo / "bilayers_extra"
    shutil.copytree(ROOT / "bilayers_extra", extras)
    shutil.copy2(ROOT / "pushdocker.cmd", repo / "pushdocker.cmd")
    (repo / "builddocker.cmd").write_text(
        '@echo off\necho Unexpected base build>"%DOCKER_CALL_LOG%"\nexit /b 97\n'
    )
    stubs = tmp_path / "stub commands"
    stubs.mkdir()
    (stubs / "docker.cmd").write_text(
        "@echo off\n"
        'echo %*>>"%DOCKER_CALL_LOG%"\n'
        'if "%~1"=="image" exit /b %DOCKER_INSPECT_EXIT%\n'
        'if "%~1"=="build" exit /b %DOCKER_BUILD_EXIT%\n'
        "exit /b 98\n"
    )
    env = os.environ.copy()
    env.update(
        PATH=str(stubs) + os.pathsep + env["PATH"],
        DOCKER_CALL_LOG=str(tmp_path / "docker-calls.log"),
        DOCKER_INSPECT_EXIT="0",
        DOCKER_BUILD_EXIT="0",
    )
    return repo, env


def run_batch(path, args, tmp_path, env):
    return subprocess.run(
        [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "call", str(path), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@WINDOWS_ONLY
@pytest.mark.parametrize("skip_build", [False, True])
@pytest.mark.parametrize("version", ["v9.8.7", "v9.8.7-beta"])
def test_publication_dry_run_never_builds_or_publishes_optional_images(
    tmp_path, skip_build, version
):
    repo, env = command_repo(tmp_path)
    (repo / "version.txt").write_text(version + "\n")
    args = ["--dry-run"] + (["--skip-build"] if skip_build else [])
    result = run_batch(repo / "pushdocker.cmd", args, tmp_path, env)
    assert result.returncode == 0, result.stdout + result.stderr
    pushes = re.findall(r'\[dry-run\] docker push "([^"]+)"', result.stdout)
    expected = [f"cellularimagingcf/w_cisegmentation:{version}"]
    if "-" not in version:
        expected.append("cellularimagingcf/w_cisegmentation:latest")
    assert pushes == expected
    assert "gradio" not in result.stdout and "jupyter" not in result.stdout
    assert not Path(env["DOCKER_CALL_LOG"]).exists()


@WINDOWS_ONLY
@pytest.mark.parametrize("variant", ["gradio", "jupyter"])
def test_manual_extra_build_uses_existing_base_and_its_own_context(tmp_path, variant):
    repo, env = command_repo(tmp_path)
    extras = repo / "bilayers_extra"
    result = run_batch(
        extras / f"builddocker_{variant}.cmd", ["--no-cache"], tmp_path, env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = Path(env["DOCKER_CALL_LOG"]).read_text().splitlines()
    assert len(calls) == 2
    assert calls[0] == 'image inspect "w_cisegmentation:v9.8.7"'
    assert calls[1].startswith("build --no-cache ")
    assert f'-f "{extras / ("Dockerfile." + variant)}"' in calls[1]
    assert f'"{extras}\\."' in calls[1]
    assert "--build-arg BASE_IMAGE=w_cisegmentation:v9.8.7" in calls[1]
    assert f"w_cisegmentation:v9.8.7-{variant}" in calls[1]
    assert "push" not in calls[1]


@WINDOWS_ONLY
@pytest.mark.parametrize("variant", ["gradio", "jupyter"])
def test_manual_extra_build_requires_base_and_propagates_build_errors(
    tmp_path, variant
):
    repo, env = command_repo(tmp_path)
    script = repo / "bilayers_extra" / f"builddocker_{variant}.cmd"
    env["DOCKER_INSPECT_EXIT"] = "1"
    result = run_batch(script, [], tmp_path, env)
    assert result.returncode == 1
    assert "Build the workflow image first" in result.stdout
    assert len(Path(env["DOCKER_CALL_LOG"]).read_text().splitlines()) == 1
    env["DOCKER_INSPECT_EXIT"] = "0"
    env["DOCKER_BUILD_EXIT"] = "17"
    result = run_batch(script, [], tmp_path, env)
    assert result.returncode == 17


@WINDOWS_ONLY
@pytest.mark.parametrize("variant", ["gradio", "jupyter"])
@pytest.mark.parametrize("publishing_arg", ["--push", "--output=type=registry"])
def test_manual_extra_build_rejects_publishing_options(
    tmp_path, variant, publishing_arg
):
    repo, env = command_repo(tmp_path)
    result = run_batch(
        repo / "bilayers_extra" / f"builddocker_{variant}.cmd",
        [publishing_arg],
        tmp_path,
        env,
    )
    assert result.returncode == 1
    assert "local builds only" in result.stdout
    assert not Path(env["DOCKER_CALL_LOG"]).exists()


@WINDOWS_ONLY
@pytest.mark.parametrize("variant", ["gradio", "jupyter"])
def test_manual_extra_build_rejects_empty_version_despite_parent_environment(
    tmp_path, variant
):
    repo, env = command_repo(tmp_path)
    (repo / "version.txt").write_text("")
    env["VERSION"] = "v1.2.3"
    result = run_batch(
        repo / "bilayers_extra" / f"builddocker_{variant}.cmd", [], tmp_path, env
    )
    assert result.returncode == 1
    assert "version.txt is empty" in result.stdout
    assert not Path(env["DOCKER_CALL_LOG"]).exists()
