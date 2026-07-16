# Repository Working Instructions

## Docker builds

- Test code changes locally by default using the `cisegmentation` Conda environment.
- On this workstation, invoke local Python and pytest explicitly with `C:\Users\p000881\AppData\Local\miniconda3\envs\cisegmentation\python.exe` (or use `conda run -n cisegmentation`). Do not rely on the shell's unqualified `python`, because it may resolve to the Miniconda base environment with incompatible packages such as Zarr v3.
- Do not build or rebuild any Docker image after code changes unless the user explicitly asks for a Docker build in the current request.
- A request to implement, test, or verify a change does not implicitly authorize a Docker build.
- When a change also affects Docker execution, report that the existing image does not yet contain the change and wait for explicit permission to rebuild it.
