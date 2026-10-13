# Development and testing

## Repository layout

| Path | Purpose |
| --- | --- |
| `cisegmentation/` | Inference adapters, pipeline, resource discovery, streaming, measurements, geometry and tracking |
| `wrapper.py`, `bilayers_cli.py`, `config.yaml` | Container entrypoint, descriptor parsing and workflow contract |
| `launcher.py`, `launch.cmd`, `requirements_launcher.txt` | Local PyQt frontend |
| `tests/`, `tests/data/` | Regression tests and small committed OME-Zarr fixtures |
| `tools/` | Model preparation, build-cache identities, public-data preparation and opt-in integration experiments |
| `skills/`, `plugin.json` | Portable agent skills and consumer matching metadata |
| `docs/` | Detailed user, scientific, development and release guides |
| `bundled_models/` | Pinned custom StarDist sources; not disposable test artifacts |
| `models/` | Git-ignored downloaded and validated runtime checkpoints |
| `inputfolder/`, `outputfolder/` | Local launcher defaults; generated contents are ignored |
| `bilayers_extra/` | Optional manual Gradio/Jupyter builds |

Keep live tests in `tests/` and expensive experiment runners in `tools/`.
`tools/test_*.py` are explicit programs, not part of default pytest discovery.
There is no archived test suite to remove. Compatibility tests exercise flags,
settings and checkpoints still supported by the current implementation; remove
them only when that compatibility is intentionally removed.

The fixtures stage fresh inputs and clean only `tests/inputfolder` and
`tests/outputfolder`. Large inputs, downloaded models, experiment reports and
build state are ignored. Preserve input data and evidence needed to reproduce
failures. Optional UI dependencies stay outside the headless requirements.

## Test layers

Use the `cisegmentation` Conda environment, not an unqualified base Python.
`create_env.cmd` installs the workflow, launcher and development dependencies.

```bat
test.cmd -m "not gpu and not omero"
```

Or on any platform with that environment activated:

```sh
python -m pytest -m "not gpu and not omero"
```

The CPU suite covers descriptors, versions and skills; calibrated image I/O;
measurements and optional extensions; bounded processing; simulated resource
limits and OOM recovery; launcher behavior; and publication-script contracts.
Windows batch execution tests are skipped on other platforms. Some script tests
inspect guard clauses; simulated Docker commands also verify actual dry-run
behavior without building or publishing an image.

Real inference checks are marked `gpu`; service round trips are marked `omero`.
Run them only with the required checkpoints and configured services:

```bat
test.cmd -m gpu
test.cmd -m omero
```

For all-checkpoint GPU experiments, see
[the model matrix runner](../tools/ALL_MODELS_TESTS.md). For tiled/untiled
comparisons and constrained-memory experiments, see
[large inputs](large-inputs.md) and [resource checks](slurm-resource-checks.md).
Successful execution, agreement with another run, and biological accuracy are
different outcomes. Record what each experiment actually establishes.

## Continuous integration

`.github/workflows/agent-plugin.yml` validates the plugin, runs the CPU suite
on Linux and builds a lightweight headless application layer. That build uses
a small Python base with checkpoint installation disabled. It tests Dockerfile
assembly, not production CUDA inference or offline checkpoint completeness.
Production publication uses validated model/runtime caches instead.

Before publishing, run local checks and review CI for the exact release commit.
Follow [releasing](releasing.md) for version synchronization and publication.
Follow [AGENTS.md](../AGENTS.md) for repository working rules.
