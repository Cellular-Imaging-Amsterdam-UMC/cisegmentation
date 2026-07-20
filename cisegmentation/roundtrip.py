from __future__ import annotations

import configparser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import queue
import time
from typing import Any, Callable, Iterable

from .settings import normalize_legacy_workflow_values


WORKFLOW = "cisegmentation"
DEFAULT_BIOMERO_ROOT = Path(r"E:\NL-BIOMERO")
DEFAULT_IMPORTER = "deployment_scenarios-biomero-importer-1"
DEFAULT_SERVER = "deployment_scenarios-omeroserver-1"
DEFAULT_WORKER = "deployment_scenarios-biomeroworker-1"
DEFAULT_SLURM = "slurmctld"
DOCKER_BUILD_STATE = ".docker-build-state.json"


class BiomeroWorkflowLogCompactor:
    """Deduplicate the overlapping Slurm log tails emitted by BIOMERO polling."""

    _logged_message = re.compile(
        r"^\d{4}-\d{2}-\d{2} .*?\[[^]]+\] \[\d+\] \(MainThread\)\s*(.*)$"
    )

    def __init__(self) -> None:
        self._inside_tail = False
        self._seen_tail_lines: set[str] = set()

    @classmethod
    def _payload(cls, line: str) -> str:
        prefix_free = re.sub(r"^\s*\*\s?", "", line.rstrip("\r\n"))
        logged = cls._logged_message.match(prefix_free)
        return logged.group(1) if logged else prefix_free

    def __call__(self, line: str) -> str | None:
        if "tail -n 10" in line and re.search(r'omero-(?:%j|\d+)\.log', line):
            self._inside_tail = True
            return None
        if not self._inside_tail:
            return line
        if "Issue with extracting progress:" in line:
            self._inside_tail = False
            return None
        if any(
            marker in line
            for marker in (
                "Getting status of [",
                "Retrieving a list of completed jobs",
                "Running import script",
            )
        ):
            self._inside_tail = False
            return line

        payload = self._payload(line).strip()
        if not payload or payload in self._seen_tail_lines:
            return None
        self._seen_tail_lines.add(payload)
        return f"\t* {payload}\n"


def docker_source_id(root: str | Path) -> str:
    root = Path(root).resolve()
    marker = root / "models" / ".complete.json"
    inputs = [
        root / "Dockerfile",
        root / "Dockerfile.models",
        root / "requirements.txt",
        root / "config.yaml",
        root / "wrapper.py",
        root / "bilayers_cli.py",
        root / "tools" / "download_models.py",
        root / "tools" / "cuda_smoke.py",
    ]
    inputs += sorted((root / "bundled_models").rglob("*"))
    inputs.append(marker)
    inputs = [path for path in inputs if path.is_file()]
    inputs += sorted(
        path
        for path in (root / "cisegmentation").rglob("*")
        if path.is_file()
        and path.name != "roundtrip.py"
        and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for path in inputs:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def docker_build_state_matches(
    state: dict[str, Any] | None, source_id: str, image_id: str
) -> bool:
    return bool(
        state
        and state.get("source_id") == source_id
        and state.get("image_id") == image_id
    )


def read_docker_build_state(root: str | Path) -> dict[str, Any] | None:
    path = Path(root) / DOCKER_BUILD_STATE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def write_docker_build_state(
    root: str | Path, source_id: str, image_id: str, image: str
) -> None:
    path = Path(root) / DOCKER_BUILD_STATE
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(
            {
                "schema": 1,
                "source_id": source_id,
                "image_id": image_id,
                "image": image,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def first_ome_zarr(folder: str | Path) -> Path:
    root = Path(folder)
    stores = sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.lower().endswith(".ome.zarr")),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not stores:
        raise FileNotFoundError(f"No top-level .ome.zarr store found in {root}")
    return stores[0]


def is_hcs_store(store: str | Path) -> bool:
    attrs = Path(store) / ".zattrs"
    if not attrs.exists():
        return False
    try:
        return isinstance(json.loads(attrs.read_text(encoding="utf-8")).get("plate"), dict)
    except (OSError, ValueError):
        return False


def redact(text: str, secrets: Iterable[str]) -> str:
    result = text
    for secret in secrets:
        if not secret:
            continue
        if len(secret) <= 5 and secret.isalnum():
            escaped = re.escape(secret)
            result = re.sub(
                rf"(?i)(OMERO_(?:ROOT_)?PASSWORD=){escaped}\b",
                r"\1<redacted>",
                result,
            )
            result = re.sub(rf"(?i)(\s-w\s+){escaped}\b", r"\1<redacted>", result)
        else:
            result = result.replace(secret, "<redacted>")
    return result


def descriptor_url(commit: str) -> str:
    return (
        "https://github.com/Cellular-Imaging-Amsterdam-UMC/"
        f"cisegmentation/tree/{commit}/config.yaml"
    )


def _json_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return [str(item) for item in value]
    except ValueError:
        pass
    return [item.strip().strip('"') for item in raw.strip("[]").split(",") if item.strip()]


def update_biomero_config(text: str, repo_url: str) -> tuple[str, bool]:
    """Update only cisegmentation keys while preserving the commented INI file."""
    lines = text.splitlines()
    section_starts: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^\s*\[([^]]+)]", line)
        if match:
            section_starts[match.group(1).upper()] = index

    workflow_section = "WORKFLOWS" if "WORKFLOWS" in section_starts else "MODELS"
    if workflow_section not in section_starts or "UI" not in section_starts:
        raise ValueError("slurm-config.ini must contain [WORKFLOWS] (or [MODELS]) and [UI]")

    def section_end(section: str) -> int:
        start = section_starts[section]
        return next(
            (i for i in range(start + 1, len(lines)) if re.match(r"^\s*\[([^]]+)]", lines[i])),
            len(lines),
        )

    wanted = {
        WORKFLOW: WORKFLOW,
        f"{WORKFLOW}_use_gpu": "true",
        f"{WORKFLOW}_job_partition": "gpu",
        f"{WORKFLOW}_job_gpus": "1",
        f"{WORKFLOW}_repo": repo_url,
        f"{WORKFLOW}_job": f"jobs/{WORKFLOW}.sh",
    }
    start, end = section_starts[workflow_section], section_end(workflow_section)
    found: set[str] = set()
    for index in range(start + 1, end):
        match = re.match(r"^\s*([^#;][^=]*?)\s*=", lines[index])
        if match and match.group(1).strip() in wanted:
            key = match.group(1).strip()
            lines[index] = f"{key} = {wanted[key]}"
            found.add(key)
    additions = [f"{key} = {value}" for key, value in wanted.items() if key not in found]
    if additions:
        lines[end:end] = ["", "# CI Segmentation local roundtrip", *additions]

    # Recompute UI bounds because workflow insertions shift it.
    ui_start = next(i for i, line in enumerate(lines) if re.match(r"^\s*\[UI]", line, re.I))
    ui_end = next(
        (i for i in range(ui_start + 1, len(lines)) if re.match(r"^\s*\[([^]]+)]", lines[i])),
        len(lines),
    )
    for key in ("zarr_workflows", "plate_workflows"):
        key_index = next(
            (i for i in range(ui_start + 1, ui_end) if re.match(rf"^\s*{key}\s*=", lines[i])),
            None,
        )
        values: list[str] = []
        if key_index is not None:
            values = _json_list(lines[key_index].split("=", 1)[1].strip())
        if WORKFLOW not in values:
            values.append(WORKFLOW)
        rendered = f"{key} = {json.dumps(values, separators=(',', ':'))}"
        if key_index is None:
            lines.insert(ui_end, rendered)
            ui_end += 1
        else:
            lines[key_index] = rendered

    updated = "\n".join(lines) + ("\n" if text.endswith(("\n", "\r")) else "")
    return updated, updated != text.replace("\r\n", "\n")


def image_manifest_matches(current: dict[str, Any], remote: dict[str, Any] | None) -> bool:
    if not remote:
        return False
    keys = ("manifest_version", "docker_image_id", "descriptor_sha256", "git_commit", "image_tag")
    return all(remote.get(key) == current.get(key) for key in keys)


def roundtrip_command(
    python: str,
    script: str | Path,
    input_dir: str,
    output_dir: str,
    parameters: dict[str, Any],
    gpu: bool,
) -> list[str]:
    return [
        python,
        str(script),
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--parameters-json",
        json.dumps(parameters, separators=(",", ":")),
        "--gpu",
        "true" if gpu else "false",
    ]


class CommandError(RuntimeError):
    pass


class RoundtripRunner:
    def __init__(
        self,
        root: Path,
        input_dir: Path,
        output_dir: Path,
        parameters: dict[str, Any],
        *,
        gpu: bool = True,
        biomero_root: Path = DEFAULT_BIOMERO_ROOT,
        timeout: int = 6 * 60 * 60,
        emit: Callable[[str], None] = print,
    ):
        parameters = normalize_legacy_workflow_values(parameters)
        self.root = root.resolve()
        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.parameters = parameters
        self.gpu = gpu
        self.biomero_root = biomero_root
        self.timeout = timeout
        self.emit = emit
        self.started_monotonic = time.monotonic()
        self.run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        self.log_dir = self.output_dir / "logs" / self.run_id
        self.summary: dict[str, Any] = {
            "roundtrip_id": self.run_id,
            "status": "running",
            "parameters": parameters,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "temporary_omero": {},
            "result_paths": [],
        }
        self.password = self._password()
        self.cancelled = False
        self.slurm_job_id: str | None = None
        self.process_id: str | None = None

    def _emit(self, line: str) -> None:
        """Emit progress without allowing a console codec to abort the run.

        Command logs are always written as UTF-8.  The fallback only affects the
        live Windows console/QProcess display when it uses a legacy code page.
        """
        try:
            self.emit(line)
        except UnicodeEncodeError:
            self.emit(line.encode("ascii", errors="replace").decode("ascii"))

    def _password(self) -> str:
        if os.environ.get("OMERO_ROOT_PASSWORD"):
            return os.environ["OMERO_ROOT_PASSWORD"]
        env_file = self.biomero_root / "deployment_scenarios" / "linux.env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("OMERO_ROOT_PASSWORD="):
                    return line.split("=", 1)[1].strip().strip('"')
        return "omero"

    def phase(self, name: str) -> None:
        self.summary["phase"] = name
        self._emit(f"PHASE: {name}")
        self._write_summary()

    def run_cmd(
        self,
        command: list[str],
        log_name: str,
        *,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
        heartbeat: str | None = None,
        on_line: Callable[[str], None] | None = None,
        echo_output: bool = True,
        output_filter: Callable[[str], str | None] | None = None,
        raw_log_name: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        safe = redact(subprocess.list2cmdline(command), [self.password])
        if echo_output:
            self._emit(f"$ {safe}")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        proc = subprocess.Popen(
            command,
            cwd=self.root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=merged_env,
            bufsize=1,
        )
        lines: list[str] = []
        pending: queue.Queue[str | None] = queue.Queue()

        def reader() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    pending.put(line)
            finally:
                pending.put(None)

        threading.Thread(target=reader, daemon=True).start()
        command_started = time.monotonic()
        deadline = command_started + timeout if timeout else None
        log_path = self.log_dir / log_name
        raw_log_file = (
            (self.log_dir / raw_log_name).open("w", encoding="utf-8")
            if raw_log_name
            else None
        )
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                if raw_log_file:
                    raw_log_file.write(f"$ {safe}\n\n")
                log_file.write(f"$ {safe}\n\n")
                finished = False
                last_heartbeat = time.monotonic()
                while not finished:
                    if deadline is not None and time.monotonic() > deadline:
                        proc.kill()
                        raise subprocess.TimeoutExpired(command, timeout)
                    try:
                        line = pending.get(timeout=0.2)
                    except queue.Empty:
                        if heartbeat and time.monotonic() - last_heartbeat >= 30:
                            elapsed = int(time.monotonic() - command_started)
                            status = f"{heartbeat} ({elapsed}s elapsed)"
                            log_file.write(f"# {status}\n")
                            log_file.flush()
                            if raw_log_file:
                                raw_log_file.write(f"# {status}\n")
                                raw_log_file.flush()
                            if echo_output:
                                self._emit(status)
                            last_heartbeat = time.monotonic()
                        if proc.poll() is not None and pending.empty():
                            break
                        continue
                    if line is None:
                        finished = True
                        continue
                    safe_line = redact(line, [self.password])
                    lines.append(safe_line)
                    if raw_log_file:
                        raw_log_file.write(safe_line)
                        raw_log_file.flush()
                    visible_line = (
                        output_filter(safe_line) if output_filter else safe_line
                    )
                    if visible_line is not None:
                        log_file.write(visible_line)
                        log_file.flush()
                        if echo_output:
                            self._emit(visible_line.rstrip("\r\n"))
                    if on_line:
                        on_line(safe_line)
        finally:
            if raw_log_file:
                raw_log_file.close()
        return_code = proc.wait()
        output = "".join(lines)
        if return_code:
            raise CommandError(f"Command failed ({return_code}): {safe}")
        return subprocess.CompletedProcess(command, return_code, output, "")

    def _git(self) -> tuple[str, str]:
        commit = self.run_cmd(["git", "rev-parse", "HEAD"], "git.log").stdout.strip()
        config_dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", "config.yaml"], cwd=self.root
        ).returncode
        if config_dirty:
            raise CommandError("config.yaml has uncommitted changes; commit and push it before a BIOMERO roundtrip")
        remote = self.run_cmd(
            ["git", "ls-remote", "origin", commit], "git_remote.log", timeout=60
        ).stdout
        if commit not in remote:
            # ls-remote <sha> is not supported by every server; inspect remote branches too.
            branches = self.run_cmd(
                ["git", "branch", "-r", "--contains", commit], "git_remote_contains.log"
            ).stdout
            if "origin/" not in branches:
                raise CommandError(f"Git commit {commit} is not reachable from origin")
        dirty = self.run_cmd(["git", "status", "--short"], "git_status.log").stdout
        return commit, dirty

    def _build(self) -> dict[str, Any]:
        version = (self.root / "version.txt").read_text(encoding="utf-8").strip() or "0.1.0"
        self.run_cmd(
            [sys.executable, str(self.root / "tools" / "download_models.py"), "--models-dir", str(self.root / "models")],
            "model_cache.log",
            timeout=self.timeout,
        )
        marker = self.root / "models" / ".complete.json"
        cache_id = hashlib.sha256(marker.read_bytes()).hexdigest()
        cache_image = f"w_cisegmentation-model-cache:{cache_id}"
        installed = subprocess.run(
            ["docker", "image", "inspect", cache_image],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        if installed.returncode != 0:
            self.run_cmd(
                ["docker", "build", "-f", "Dockerfile.models", "--build-arg", f"MODEL_CACHE_ID={cache_id}", "-t", cache_image, "-t", "w_cisegmentation-model-cache:latest", "."],
                "docker_model_cache_build.log",
                timeout=self.timeout,
            )
        source_id = docker_source_id(self.root)
        image_name = "w_cisegmentation:latest"
        inspect = subprocess.run(
            ["docker", "image", "inspect", image_name],
            cwd=self.root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        metadata = json.loads(inspect.stdout)[0] if inspect.returncode == 0 else None
        if metadata and docker_build_state_matches(
            read_docker_build_state(self.root),
            source_id,
            str(metadata.get("Id", "")),
        ):
            self._emit(
                "Local Docker image matches the current workflow source; build skipped."
            )
            self.summary["docker_build_reused"] = True
            return {
                "version": version,
                "tag": "latest",
                "image_id": str(metadata.get("Id", "")),
                "content_id": source_id,
            }
        self.run_cmd(
            ["docker", "build", "--build-arg", f"MODEL_CACHE_IMAGE={cache_image}", "-t", f"w_cisegmentation:{version}", "-t", "w_cisegmentation:latest", "."],
            "docker_build.log",
            timeout=self.timeout,
        )
        inspect = subprocess.run(
            ["docker", "image", "inspect", image_name],
            cwd=self.root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=True,
        )
        metadata = json.loads(inspect.stdout)[0]
        image_id = str(metadata.get("Id", ""))
        write_docker_build_state(self.root, source_id, image_id, image_name)
        self.summary["docker_build_reused"] = False
        return {
            "version": version,
            "tag": "latest",
            "image_id": image_id,
            "content_id": source_id,
        }

    def _register(self, commit: str) -> bool:
        config_path = self.biomero_root / "web" / "slurm-config.ini"
        original = config_path.read_text(encoding="utf-8")
        updated, changed = update_biomero_config(original, descriptor_url(commit))
        if changed:
            backup = config_path.with_name(config_path.name + f".bak.{self.run_id}")
            shutil.copy2(config_path, backup)
            config_path.write_text(updated, encoding="utf-8", newline="\n")
            shutil.copy2(backup, self.log_dir / "slurm-config.backup.ini")
            self.summary["config_backup"] = str(backup)
        return changed

    def _regenerate_job(self) -> None:
        code = (
            "from biomero import SlurmClient\n"
            "w='cisegmentation'\n"
            "SlurmClient.init_workflows=lambda self,force_update=False: None\n"
            "with SlurmClient.from_config(configfile='/opt/omero/server/slurm-config.ini', init_slurm=False) as c:\n"
            "  for a in ('slurm_model_repos','slurm_model_jobs','slurm_model_images'):\n"
            "    m=getattr(c,a,None) or {}; setattr(c,a,{w:m.get(w, 'jobs/cisegmentation.sh')})\n"
            "  c.setup_directories(); c.update_slurm_scripts(generate_jobs=True)\n"
        )
        self.run_cmd(
            ["docker", "exec", "-i", DEFAULT_WORKER, "/opt/omero/server/venv-3.11/bin/python", "-c", code],
            "biomero_setup.log",
            timeout=600,
        )

    def _sync_sif(self, identity: dict[str, Any]) -> None:
        remote_dir = "/data/my-scratch/singularity_images/workflows/cisegmentation"
        sif = f"{remote_dir}/w_cisegmentation_latest.sif"
        manifest_path = f"{sif}.manifest.json"
        remote_manifest: dict[str, Any] | None = None
        read = subprocess.run(
            ["docker", "exec", DEFAULT_SLURM, "cat", manifest_path], text=True, capture_output=True
        )
        if read.returncode == 0:
            try:
                remote_manifest = json.loads(read.stdout)
            except ValueError:
                pass
        exists = subprocess.run(
            ["docker", "exec", DEFAULT_SLURM, "test", "-s", sif]
        ).returncode == 0
        if exists and image_manifest_matches(identity, remote_manifest):
            self._emit("Slurm SIF already matches the local Docker image; conversion skipped.")
            self.summary["sif_reused"] = True
            return
        if (
            exists
            and remote_manifest
            and not remote_manifest.get("manifest_version")
            and all(remote_manifest.get(key) == identity.get(key) for key in ("descriptor_sha256", "git_commit", "image_tag"))
        ):
            migrate = self.run_cmd(
                ["docker", "exec", DEFAULT_SLURM, "bash", "-lc", f"tool=$(command -v apptainer || command -v singularity); $tool exec {shlex.quote(sif)} python /app/wrapper.py --help >/dev/null && sha256sum {shlex.quote(sif)}"],
                "sif_manifest_migration.log",
                timeout=600,
            ).stdout
            checksum = migrate.split()[-2] if len(migrate.split()) >= 2 else ""
            migrated = {**identity, "sif_sha256": checksum}
            self.run_cmd(
                ["docker", "exec", DEFAULT_SLURM, "bash", "-lc", f"printf %s {shlex.quote(json.dumps(migrated, sort_keys=True))} > {shlex.quote(manifest_path)}"],
                "sif_manifest.log",
            )
            self.summary["sif_reused"] = True
            self.summary["sif_manifest_migrated"] = True
            self.summary["slurm_image"] = sif
            return

        resume = self.run_cmd(
            [
                "docker", "exec", DEFAULT_SLURM, "bash", "-lc",
                f"tool=$(command -v apptainer || command -v singularity); candidate=$(ls -t {shlex.quote(sif)}.partial.* {shlex.quote(sif)}.partial 2>/dev/null | head -1 || true); if [ -n \"$candidate\" ] && $tool exec \"$candidate\" python /app/wrapper.py --help >/dev/null; then mv \"$candidate\" {shlex.quote(sif)}; echo RESUMED; else echo NONE; fi",
            ],
            "sif_resume.log",
            timeout=600,
        ).stdout
        if "RESUMED" in resume:
            checksum = self.run_cmd(
                ["docker", "exec", DEFAULT_SLURM, "sha256sum", sif],
                "sif_checksum.log",
                timeout=self.timeout,
            ).stdout.split()[0]
            resumed_identity = {**identity, "sif_sha256": checksum}
            self.run_cmd(
                ["docker", "exec", DEFAULT_SLURM, "bash", "-lc", f"printf %s {shlex.quote(json.dumps(resumed_identity, sort_keys=True))} > {shlex.quote(manifest_path)}"],
                "sif_manifest.log",
            )
            self.summary["sif_reused"] = False
            self.summary["sif_resumed"] = True
            self.summary["slurm_image"] = sif
            return

        archive_name = f"w_cisegmentation_{identity['docker_image_id'].split(':')[-1][:12]}.tar"
        remote_archive = f"{remote_dir}/archives/{archive_name}"
        self.run_cmd(
            ["docker", "exec", DEFAULT_SLURM, "mkdir", "-p", f"{remote_dir}/archives"],
            "sif_prepare.log",
        )
        volume = self.run_cmd(
            ["docker", "inspect", DEFAULT_SLURM, "--format", '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}'],
            "slurm_volume.log",
        ).stdout.strip()
        if not volume:
            raise CommandError("Could not identify the Slurm /data Docker volume")
        volume_archive = remote_archive.removeprefix("/data/")
        self.run_cmd(
            [
                "docker", "run", "--rm",
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-v", f"{volume}:/slurm-data",
                "docker:cli",
                "sh", "-c",
                f"rm -f /slurm-data/{volume_archive}.partial && docker image save -o /slurm-data/{volume_archive}.partial w_cisegmentation:latest && mv /slurm-data/{volume_archive}.partial /slurm-data/{volume_archive}",
            ],
            "docker_save.log",
            timeout=self.timeout,
        )
        self.run_cmd(
            ["docker", "exec", DEFAULT_SLURM, "bash", "-lc", f"tool=$(command -v apptainer || command -v singularity) && candidate=$(ls -t {shlex.quote(sif)}.partial.* {shlex.quote(sif)}.partial 2>/dev/null | head -1 || true) && if [ -n \"$candidate\" ] && $tool exec \"$candidate\" python /app/wrapper.py --help >/dev/null; then tmp=\"$candidate\"; else tmp={shlex.quote(sif)}.partial.$$; rm -f \"$tmp\"; $tool build --force \"$tmp\" docker-archive://{shlex.quote(remote_archive)}; $tool exec \"$tmp\" python /app/wrapper.py --help >/dev/null; fi && mv \"$tmp\" {shlex.quote(sif)}"],
            "sif_build.log",
            timeout=self.timeout,
        )
        checksum = self.run_cmd(
            ["docker", "exec", DEFAULT_SLURM, "sha256sum", sif], "sif_checksum.log", timeout=self.timeout
        ).stdout.split()[0]
        identity = {**identity, "sif_sha256": checksum}
        manifest_json = json.dumps(identity, sort_keys=True)
        self.run_cmd(
            ["docker", "exec", DEFAULT_SLURM, "bash", "-lc", f"printf %s {shlex.quote(manifest_json)} > {shlex.quote(manifest_path)}"],
            "sif_manifest.log",
        )
        self.summary["sif_reused"] = False
        self.summary["slurm_image"] = sif

    def _container_python(self, code: str, env: dict[str, str], log: str) -> str:
        command = ["docker", "exec"]
        for key, value in env.items():
            command += ["-e", f"{key}={value}"]
        command += [DEFAULT_IMPORTER, "python", "-c", code]
        return self.run_cmd(command, log, timeout=600).stdout.strip().splitlines()[-1]

    def _create_container(self, kind: str, name: str) -> int:
        model = "ScreenI" if kind == "Screen" else "DatasetI"
        code = f"""
import os
from omero.gateway import BlitzGateway
from omero.model import {model}
from omero.rtypes import rstring
c=BlitzGateway('root',os.environ['OMERO_PASSWORD'],host='omeroserver',port=4064); assert c.connect()
g=c.getObject('ExperimenterGroup',attributes={{'name':'system'}}); c.setGroupForSession(g.getId())
o={model}(); o.setName(rstring(os.environ['OBJECT_NAME'])); o=c.getUpdateService().saveAndReturnObject(o); print(o.getId().getValue()); c.close()
"""
        return int(self._container_python(code, {"OMERO_PASSWORD": self.password, "OBJECT_NAME": name}, f"create_{kind.lower()}.log"))

    def _probe(
        self, args: list[str], log: str, *, retain_staging: bool = False
    ) -> dict[str, Any]:
        probe = self.root / "tools" / "omero_import_metadata_probe" / "omero_import_metadata_probe.py"
        if not probe.exists():
            probe = Path(r"C:\rahoebe\Python\cideconvolve\tools\omero_import_metadata_probe\omero_import_metadata_probe.py")
        report = self.log_dir / (Path(log).stem + "_report")
        command = [sys.executable, str(probe), "--importer-container", DEFAULT_IMPORTER, "--user", "root", "--group", "system", *args]
        if retain_staging:
            command.append("--retain-container-staging")
        command += ["--out", str(report)]
        self.run_cmd(command, log, timeout=self.timeout, env={"OMERO_ROOT_PASSWORD": self.password})
        return json.loads((report / "report.json").read_text(encoding="utf-8"))

    @staticmethod
    def _imported_ids(report: dict[str, Any], kind: str) -> list[int]:
        key = "plates" if kind == "Plate" else "images"
        ids: list[int] = []
        for result in report.get("imports", {}).values():
            ids += [int(item["id"]) for item in result.get("objects", {}).get(key, [])]
        return ids

    def _workflow_args(self, data_type: str, object_id: int, target_kind: str, target_id: int) -> list[str]:
        args = [
            "Data_Type=" + data_type,
            f"IDs={object_id}",
            "Use_ZARR_Format=true",
            "OME-Zarr_version=0.4",
            "Select how to import your results (one or more)=true",
            "Cleanup?=false",
            f"{WORKFLOW}=true",
            f"{WORKFLOW}_Version=latest",
        ]
        if target_kind == "Screen":
            args += [f"Screen_ID={target_id}", "3a) Import into NEW Screen=roundtrip-output"]
        else:
            args += [f"Dataset_ID={target_id}", "3a) Import into NEW Dataset=roundtrip-output"]
        for key, value in self.parameters.items():
            if value in (None, "", []):
                continue
            if isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, list):
                value = ",".join(map(str, value))
            args.append(f"{WORKFLOW}_|_{key}={value}")
        return args

    def _launch_workflow(self, data_type: str, object_id: int, target_kind: str, target_id: int) -> None:
        script_id = "2560" if data_type in ("Plate", "Screen") else "2561"
        cli = "/opt/omero/server/venv-3.11/bin/omero"
        command = [
            "docker", "exec", "-e", f"OMERO_PASSWORD={self.password}", DEFAULT_SERVER,
            cli, "-s", "localhost", "-u", "root", "-g", "system", "script", "launch", script_id,
            *self._workflow_args(data_type, object_id, target_kind, target_id),
        ]
        def capture_identifiers(line: str) -> None:
            execution = re.search(
                r"(?:execution(?:[_ ]id)?|workflow(?:[_ ]id)?)\D+"
                r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
                line,
                re.I,
            )
            process = re.search(r"\bJob\s+(\d+)\s+ready\b", line, re.I) or re.search(
                r"\bprocess[_ ]id\D+(\d+)", line, re.I
            )
            job = re.search(r"\bSubmitted\b.*?\bas batch job\s+(\d+)\b", line, re.I) or re.search(
                r"\bSubmitted batch job\s+(\d+)\b", line, re.I
            )
            if execution and "biomero_execution_id" not in self.summary:
                execution_id = execution.group(1)
                self.summary["biomero_execution_id"] = execution_id
                self._emit(f"BIOMERO_EXECUTION_ID: {execution_id}")
            if process and not self.process_id:
                self.process_id = process.group(1)
                self.summary["omero_process_id"] = self.process_id
                self._emit(f"OMERO_PROCESS_ID: {self.process_id}")
            if job and self.slurm_job_id != job.group(1):
                self.slurm_job_id = job.group(1)
                self.summary["slurm_job_id"] = self.slurm_job_id
                self._emit(f"SLURM_JOB_ID: {self.slurm_job_id}")

        proc = self.run_cmd(
            command,
            "biomero_workflow.log",
            timeout=self.timeout,
            heartbeat="Waiting for BIOMERO analysis and OMERO result import…",
            on_line=capture_identifiers,
            output_filter=BiomeroWorkflowLogCompactor(),
            raw_log_name="biomero_workflow.raw.log",
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        capture_identifiers(output)

    def _children(self, kind: str, parent_id: int) -> list[int]:
        code = """
import os
from omero.gateway import BlitzGateway
c=BlitzGateway('root',os.environ['OMERO_PASSWORD'],host='omeroserver',port=4064); assert c.connect()
g=c.getObject('ExperimenterGroup',attributes={'name':'system'}); c.setGroupForSession(g.getId())
k=os.environ['KIND']; p=int(os.environ['PARENT'])
if k=='Dataset': ids=[x.getId() for x in c.getObject('Dataset',p).listChildren()]
else:
 ids=[]
 for plate in c.getObject('Screen',p).listChildren(): ids.append(plate.getId())
print(','.join(map(str,ids))); c.close()
"""
        raw = self._container_python(code, {"OMERO_PASSWORD": self.password, "KIND": kind, "PARENT": str(parent_id)}, "query_results.log")
        return [int(item) for item in raw.split(",") if item.strip().isdigit()]

    def _apply_label_lut(self, kind: str, result_ids: list[int]) -> None:
        """Apply OMERO's black-background Glasbey LUT to imported results."""
        image_ids = result_ids
        if kind == "Screen":
            image_ids = [
                int(field["image_id"])
                for plate_id in result_ids
                for field in self._plate_fields(plate_id)
            ]
        code = """
import os
from omero.gateway import BlitzGateway
c=BlitzGateway('root',os.environ['OMERO_PASSWORD'],host='omeroserver',port=4064); assert c.connect()
g=c.getObject('ExperimenterGroup',attributes={'name':'system'}); c.setGroupForSession(g.getId())
updated=[]
for image_id in [int(value) for value in os.environ['IMAGE_IDS'].split(',') if value]:
 image=c.getObject('Image',image_id)
 if image is None: continue
 channels=image.getChannels()
 active=list(range(1,len(channels)+1))
 windows=[[channel.getWindowStart(),channel.getWindowEnd()] for channel in channels]
 image.set_active_channels(active,windows=windows,colors=['glasbey_inverted.lut']*len(channels))
 image.saveDefaults(); updated.append(image_id)
print(','.join(map(str,updated))); c.close()
"""
        updated = self._container_python(
            code,
            {
                "OMERO_PASSWORD": self.password,
                "IMAGE_IDS": ",".join(map(str, image_ids)),
            },
            "apply_glasbey_lut.log",
        )
        updated_ids = [int(value) for value in updated.split(",") if value.strip().isdigit()]
        if sorted(updated_ids) != sorted(image_ids):
            raise CommandError(
                f"Could not apply Glasbey LUT to every OMERO result image: expected {image_ids}, updated {updated_ids}"
            )
        self.summary["result_lookup_table"] = "glasbey_inverted.lut"

    def _export_results(self, kind: str, result_ids: list[int]) -> list[Path]:
        exported: list[Path] = []
        if kind == "Screen":
            for plate_id in result_ids:
                fields = self._plate_fields(plate_id)
                if not fields:
                    raise CommandError(f"OMERO Plate:{plate_id} contains no field images")
                destination = self.output_dir / f"roundtrip_{self.run_id}_plate_{plate_id}.ome.zarr"
                partial = destination.with_name(destination.name + ".partial")
                if partial.exists():
                    shutil.rmtree(partial)
                partial.mkdir(parents=True)
                (partial / ".zgroup").write_text('{"zarr_format":2}\n', encoding="utf-8")
                wells: dict[str, list[str]] = {}
                rows: set[str] = set()
                columns: set[str] = set()
                for field in fields:
                    row, column, index = str(field["row"]), str(field["column"]), str(field["field"])
                    well_path = f"{row}/{column}"
                    rows.add(row)
                    columns.add(column)
                    wells.setdefault(well_path, []).append(index)
                    report = self._probe(
                        ["--slurm-input-image", str(field["image_id"])],
                        f"result_plate_{plate_id}_image_{field['image_id']}.log",
                    )
                    source = Path(report["slurm_input_export"]["local_zarr_path"])
                    shutil.copytree(source, partial / well_path / index, dirs_exist_ok=True)
                for well_path, indexes in wells.items():
                    well = partial / well_path
                    (well / ".zgroup").write_text('{"zarr_format":2}\n', encoding="utf-8")
                    (well / ".zattrs").write_text(
                        json.dumps({"well": {"images": [{"path": item} for item in sorted(indexes, key=int)], "version": "0.4"}}, indent=2),
                        encoding="utf-8",
                    )
                plate_attrs = {
                    "plate": {
                        "version": "0.4",
                        "name": f"OMERO Plate {plate_id}",
                        "rows": [{"name": row} for row in sorted(rows)],
                        "columns": [{"name": column} for column in sorted(columns, key=int)],
                        "wells": [{"path": path} for path in sorted(wells)],
                    },
                    "cisegmentation_roundtrip": {"source_omero_plate_id": plate_id, "field_count": len(fields)},
                }
                (partial / ".zattrs").write_text(json.dumps(plate_attrs, indent=2), encoding="utf-8")
                partial.replace(destination)
                exported.append(destination)
            return exported
        for image_id in result_ids:
            report = self._probe(["--slurm-input-image", str(image_id)], f"result_export_{image_id}.log")
            source = Path(report["slurm_input_export"]["local_zarr_path"])
            destination = self.output_dir / f"roundtrip_{self.run_id}_{source.name}"
            partial = destination.with_name(destination.name + ".partial")
            if partial.exists():
                shutil.rmtree(partial)
            shutil.copytree(source, partial)
            partial.replace(destination)
            exported.append(destination)
        return exported

    def _plate_fields(self, plate_id: int) -> list[dict[str, Any]]:
        code = """
import json, os
from omero.gateway import BlitzGateway
c=BlitzGateway('root',os.environ['OMERO_PASSWORD'],host='omeroserver',port=4064); assert c.connect()
g=c.getObject('ExperimenterGroup',attributes={'name':'system'}); c.setGroupForSession(g.getId())
p=c.getObject('Plate',int(os.environ['PLATE_ID'])); out=[]
for well in p.listChildren():
 row=chr(ord('A')+int(well.getRow())); col=str(int(well.getColumn())+1)
 for field,sample in enumerate(well.listChildren()):
  image=sample.getImage()
  if image: out.append({'row':row,'column':col,'field':field,'image_id':image.getId()})
print(json.dumps(out)); c.close()
"""
        raw = self._container_python(
            code,
            {"OMERO_PASSWORD": self.password, "PLATE_ID": str(plate_id)},
            f"plate_{plate_id}_fields.log",
        )
        value = json.loads(raw)
        return value if isinstance(value, list) else []

    def _collect_slurm_logs(self) -> None:
        job = self.slurm_job_id or ""
        # Restrict container logs to this roundtrip and keep their verbose,
        # potentially Unicode-rich contents in UTF-8 files instead of the UI.
        since_seconds = max(60, int(time.monotonic() - self.started_monotonic) + 60)
        commands = {
            "sacct.txt": f"sacct -j {shlex.quote(job)} --format=JobID,JobName,State,ExitCode,Elapsed,NodeList -P" if job else "sacct -S today --name cisegmentation -X -P | tail -50",
            "scontrol.txt": f"scontrol show job {shlex.quote(job)}" if job else "squeue -a",
        }
        for name, command in commands.items():
            try:
                self.run_cmd(
                    ["docker", "exec", DEFAULT_SLURM, "bash", "-lc", command],
                    name,
                    echo_output=False,
                )
            except Exception as exc:
                (self.log_dir / name).write_text(str(exc), encoding="utf-8")
        for container, name in ((DEFAULT_SERVER, "omero_server.log"), (DEFAULT_WORKER, "biomero_worker.log"), (DEFAULT_IMPORTER, "importer.log")):
            try:
                self.run_cmd(
                    ["docker", "logs", "--since", f"{since_seconds}s", container],
                    name,
                    echo_output=False,
                )
            except Exception as exc:
                (self.log_dir / name).write_text(str(exc), encoding="utf-8")

    def _delete_container(self, kind: str, object_id: int) -> None:
        code = """
import os
from omero.gateway import BlitzGateway
c=BlitzGateway('root',os.environ['OMERO_PASSWORD'],host='omeroserver',port=4064); assert c.connect()
g=c.getObject('ExperimenterGroup',attributes={'name':'system'}); c.setGroupForSession(g.getId())
k=os.environ['KIND']; i=int(os.environ['OBJECT_ID']); c.deleteObjects(k,[i],deleteAnns=True,deleteChildren=True,wait=True)
print('deleted' if c.getObject(k,i) is None else 'present'); c.close()
"""
        result = self._container_python(code, {"OMERO_PASSWORD": self.password, "KIND": kind, "OBJECT_ID": str(object_id)}, f"delete_{kind.lower()}_{object_id}.log")
        if result != "deleted":
            raise CommandError(f"OMERO {kind}:{object_id} still exists after cleanup")

    def _delete_probe_staging(self, container_path: str) -> None:
        path = PurePosixPath(container_path)
        parent = path.parent
        staging_roots = {
            PurePosixPath("/tmp/cideconvolve_omero_probe"),
            PurePosixPath("/data/.cisegmentation_roundtrip"),
        }
        if parent.parent not in staging_roots or not re.fullmatch(r"[0-9a-f]{32}", parent.name):
            raise CommandError(f"Refusing to remove unsafe probe staging path: {container_path}")
        self.run_cmd(
            [
                "docker",
                "exec",
                "-u",
                "0",
                DEFAULT_IMPORTER,
                "rm",
                "-rf",
                "--",
                str(parent),
            ],
            "delete_probe_staging.log",
            timeout=120,
        )
        exists = subprocess.run(
            ["docker", "exec", DEFAULT_IMPORTER, "test", "-e", str(parent)],
            check=False,
        ).returncode == 0
        if exists:
            raise CommandError(f"Probe staging path still exists after cleanup: {parent}")

    def _write_summary(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        safe = json.loads(redact(json.dumps(self.summary), [self.password]))
        (self.log_dir / "roundtrip.json").write_text(json.dumps(safe, indent=2), encoding="utf-8")
        lines = ["# CI Segmentation OMERO roundtrip", ""] + [f"- {key}: `{value}`" for key, value in safe.items() if not isinstance(value, (dict, list))]
        (self.log_dir / "roundtrip.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def execute(self) -> int:
        success = False
        containers: list[tuple[str, int]] = []
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            store = first_ome_zarr(self.input_dir)
            hcs = is_hcs_store(store)
            self.summary.update({"input": str(store), "is_hcs": hcs})
            self.phase("preflight and Docker image check")
            commit, dirty = self._git()
            build = self._build()
            descriptor_hash = hashlib.sha256((self.root / "config.yaml").read_bytes()).hexdigest()
            identity = {"manifest_version": 2, "docker_image_id": build["content_id"], "descriptor_sha256": descriptor_hash, "git_commit": commit, "image_tag": build["tag"]}
            self.summary.update({"git_commit": commit, "git_dirty": dirty, "docker": build, "image_identity": identity})

            self.phase("register BIOMERO workflow")
            changed = self._register(commit)
            job_exists = subprocess.run(["docker", "exec", DEFAULT_SLURM, "test", "-s", "/data/my-scratch/slurm-scripts/jobs/cisegmentation.sh"]).returncode == 0
            if changed or not job_exists:
                self._regenerate_job()

            self.phase("synchronize Slurm image")
            self._sync_sif(identity)

            parent_kind = "Screen" if hcs else "Dataset"
            data_type = "Plate" if hcs else "Image"
            stamp = f"CI segmentation roundtrip {self.run_id}"
            self.phase("create temporary OMERO containers")
            input_parent = self._create_container(parent_kind, stamp + " input")
            output_parent = self._create_container(parent_kind, stamp + " output")
            containers = [(parent_kind, input_parent), (parent_kind, output_parent)]
            self.summary["temporary_omero"] = {"kind": parent_kind, "input": input_parent, "output": output_parent}

            self.phase("import input OME-Zarr")
            import_report = self._probe(
                ["--input", str(store), "--target", f"{parent_kind}:{input_parent}", "--mode", "biomero", "--cleanup", "never"],
                "input_import.log",
                retain_staging=True,
            )
            container_input = import_report.get("container_input", {})
            staging_path = (
                str(container_input.get("container_path", ""))
                if container_input.get("method") == "docker_cp_to_tmp"
                else ""
            )
            if staging_path:
                self.summary["temporary_probe_staging"] = staging_path
            imported = self._imported_ids(import_report, data_type)
            if not imported:
                raise CommandError(f"Input import returned no OMERO {data_type} IDs")
            self.summary["input_object_ids"] = imported

            self.phase("run BIOMERO workflow")
            self._launch_workflow(data_type, imported[0], parent_kind, output_parent)

            self.phase("verify OMERO result import")
            result_ids = self._children(parent_kind, output_parent)
            if not result_ids:
                raise CommandError(f"BIOMERO completed without importing results into {parent_kind}:{output_parent}")
            self.summary["result_object_ids"] = result_ids

            benchmark = str(self.parameters.get("benchmark", False)).lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if not benchmark:
                self.phase("apply OMERO label lookup table")
                self._apply_label_lut(parent_kind, result_ids)

            self.phase("export OMERO results")
            results = self._export_results(parent_kind, result_ids)
            self.summary["result_paths"] = [str(path) for path in results]

            self.phase("collect logs")
            self._collect_slurm_logs()
            self.phase("delete temporary OMERO objects")
            for kind, object_id in reversed(containers):
                self._delete_container(kind, object_id)
            if staging_path:
                self._delete_probe_staging(staging_path)
                self.summary["temporary_probe_staging_cleanup"] = "verified"
            self.summary["cleanup"] = "verified"
            success = True
            self.summary["status"] = "success"
            return 0
        except KeyboardInterrupt:
            self.summary["status"] = "cancelled"
            self.summary["cleanup"] = "retained"
            return 130
        except Exception as exc:
            self.summary["status"] = "failed"
            self.summary["error"] = str(exc)
            self.summary["cleanup"] = "retained"
            self._emit(f"ERROR: {exc}")
            return 1
        finally:
            if not success:
                try:
                    self._collect_slurm_logs()
                except Exception:
                    pass
            self.summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._write_summary()

    def cancel(self) -> None:
        self.cancelled = True
        if self.slurm_job_id:
            subprocess.run(["docker", "exec", DEFAULT_SLURM, "scancel", self.slurm_job_id])
