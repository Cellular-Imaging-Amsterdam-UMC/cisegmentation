---
name: use-cisegmentation-workflow
description: Configure, launch, monitor, and recover the Bilayers CI Segmentation workflow for OME-Zarr images or HCS plates using Cellpose, Cellpose-SAM, StarDist, InstanSeg, or Spotiflow, including spot-only detection, spatial and colocalization measurements, cell/nucleus lineages, independent spot trajectories, and separate geometry databases.
metadata:
  version: "3"
---

# Use CI Segmentation Workflow

1. Confirm the requested OME-Zarr image or HCS plate inputs and the configured
   CI Segmentation workflow revision. Do not start execution from inspection
   alone.
2. Inspect the workflow descriptor and available CPU/GPU compute resources.
3. Read [PARAMETERS.md](references/PARAMETERS.md) before proposing settings.
4. Validate readable OME-Zarr inputs, one-based channel
   numbers, enabled segmentation steps, parameter types and ranges, compatible
   model options, output mode, and requested compute device.
5. Present the resolved workflow revision, input object, parameters, expected
   outputs, resource choice, and side effects.
6. Use the user's explicit execution request or existing authorization for
   submission. Ask for confirmation only when that authorization is absent
   or the resolved request materially changes the authorized scope.
7. Submit exactly once through the available Bilayers workflow execution
   interface. Retain the returned run or job ID.
8. Monitor that ID. Do not resubmit merely because status is delayed or a
   client response times out.
9. Read [OUTPUTS.md](references/OUTPUTS.md) to verify completion and explain
   results. Read [TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) only after
   validation, execution, or output verification fails.
10. Record the workflow key, configured ref, resolved commit, parameters, input
    stores, run ID, timestamps, final status, and discovered outputs as
    provenance.

Require user authorization before submission, cancellation, deletion, or
overwrite; an explicit request already provides that authorization. A scheduler
completion alone is not proof of successful output creation.

New measurements, tracking and geometry are opt-in. Read the extension section
in PARAMETERS.md and verify that the executable revision supports these flags.
Steps 1 and 2 may both be skipped for independent spot detection and tracking.
Never enable spot division or merging. Geometry export does not imply automatic
polygon display in an image viewer.

For retained public validation inputs, read [PUBLICDATA.md](references/PUBLICDATA.md)
before choosing reference trajectories or interpreting calibration and overlap.
