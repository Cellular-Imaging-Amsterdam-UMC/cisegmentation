# Repository Working Instructions

## Docker builds

- Default builds, CI builds, and Docker Hub publication must include only the headless workflow image. Internal model/runtime cache images remain build dependencies.
- Gradio and Jupyter Dockerfiles, build commands, and requirements belong in `bilayers_extra/`. Build these optional images only when explicitly requested, and never include them in Docker Hub publication.
- Test code changes locally by default using the `cisegmentation` Conda environment.
- On this workstation, invoke local Python and pytest explicitly with `C:\Users\p000881\AppData\Local\miniconda3\envs\cisegmentation\python.exe` (or use `conda run -n cisegmentation`). Do not rely on the shell's unqualified `python`, because it may resolve to the Miniconda base environment with incompatible packages such as Zarr v3.
- If the first-choice environment does not exist, use the fallback environment at `V:\BIOMERO-local\tests\cisegmentation`: invoke `V:\BIOMERO-local\tests\cisegmentation\python.exe` explicitly for Python and pytest, or use `conda run -p V:\BIOMERO-local\tests\cisegmentation`.
- Do not build or rebuild any Docker image after code changes unless the user explicitly asks for a Docker build in the current request.
- A request to implement, test, or verify a change does not implicitly authorize a Docker build.
- When a change also affects Docker execution, report that the existing image does not yet contain the change and wait for explicit permission to rebuild it.
