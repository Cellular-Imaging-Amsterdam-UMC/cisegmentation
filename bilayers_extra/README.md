# Optional Bilayers interfaces

This folder contains the Gradio and Jupyter Dockerfiles, requirements, and manual
Windows build scripts. These images are local optional builds. They are excluded
from default builds and CI, and `pushdocker.cmd` never publishes them to Docker
Hub.

Build the headless workflow image from the repository root first:

```bat
builddocker.cmd
```

Then explicitly build the interface you want:

```bat
bilayers_extra\builddocker_gradio.cmd
bilayers_extra\builddocker_jupyter.cmd
```

The scripts read `../version.txt`, require the matching local
`w_cisegmentation:<version>` workflow image, and use this folder as the build
context. They do not automatically rebuild the workflow image. You can invoke
them from any working directory. The optional `--no-cache` argument is supported;
publishing flags such as `--push` are rejected.

The optional local tags are `w_cisegmentation:<version>-gradio` /
`w_cisegmentation:latest-gradio` and `w_cisegmentation:<version>-jupyter` /
`w_cisegmentation:latest-jupyter`.

For manual builds on Linux, run these commands from the repository root,
replacing `<version>` with the value in `version.txt`:

```sh
docker build -f bilayers_extra/Dockerfile.gradio --build-arg BASE_IMAGE=w_cisegmentation:<version> -t w_cisegmentation:<version>-gradio bilayers_extra
docker build -f bilayers_extra/Dockerfile.jupyter --build-arg BASE_IMAGE=w_cisegmentation:<version> -t w_cisegmentation:<version>-jupyter bilayers_extra
```

Run the selected interface locally:

```sh
docker run --rm --gpus all -p 127.0.0.1:7878:7878 w_cisegmentation:<version>-gradio
docker run --rm --gpus all -p 127.0.0.1:8888:8888 w_cisegmentation:<version>-jupyter
```
