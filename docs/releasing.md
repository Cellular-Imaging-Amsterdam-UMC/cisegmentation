# Release and publication

The repository includes two Windows maintainer scripts. They contain no
credentials: Docker authentication is read from Docker's credential store and
GitHub authentication is handled by `gh auth`.

Inspect Docker tags and push commands without changing the registry:

```bat
pushdocker.cmd --skip-build --dry-run
```

Build and publish the version from `version.txt` only after reviewing the dry
run. Publication includes only the headless workflow image (version and
`latest` tags); optional Gradio and Jupyter images are no longer published:

```bat
pushdocker.cmd --yes
```

Validate a GitHub release without creating a tag or release:

```bat
release_github.cmd --dry-run
```

Create the release after reviewing the dry run:

```bat
release_github.cmd --yes
```

Publication requires a semantic version, optionally prefixed with `v` (for
example, `v0.3.0`). Docker publishing is locked to
`cellularimagingcf/w_cisegmentation`; GitHub publishing is locked to the
`Cellular-Imaging-Amsterdam-UMC/cisegmentation` origin. The GitHub release tool
also requires a clean branch that is synchronized with its upstream. If the
Docker tag in `config.yaml` differs from `version.txt`, the confirmed release
updates it, commits and pushes that synchronization, and then tags the resulting
commit. A dry run only prints those actions.

## Version contract and release checks

Stable 0.7.0 uses Git tag and Docker tag `v0.7.0`. Python packaging,
`cisegmentation.__version__` and the plugin manifest use `0.7.0`.
Update all five version locations together: `version.txt`, `config.yaml`,
`pyproject.toml`, `cisegmentation/__init__.py` and `plugin.json`.
The synchronization test accepts future versions and checks their agreement.

1. Review [development and testing](development.md), run the CPU suite and
   relevant GPU checks, and document any checks not executed.
2. Commit and push the reviewed changes; inspect the source diff and version
   agreement before release. Do not overwrite an existing release tag.
3. Build the headless image using the validated model/runtime cache described
   in [models](models.md). Run a container version/configuration check and GPU
   smoke test when CUDA is available. Lightweight CI builds do not validate
   bundled checkpoints or production CUDA execution.
4. Publish the versioned image and stable `latest`, then create the matching
   annotated Git tag and GitHub release. A prerelease publishes only its
   explicit Docker tag and leaves stable `latest` unchanged.
5. Verify the remote tag points to the reviewed commit and the release is
   published. Verify versioned and `latest` Docker manifests identify the
   intended image, and retain its digest in the release notes.

If authentication, build, tests or publication fail, retain the prepared
changes and report which stage failed. Do not describe a partial publication
as a complete release. GitHub and Docker credentials stay in their normal
credential stores; they are never committed to the repository.
