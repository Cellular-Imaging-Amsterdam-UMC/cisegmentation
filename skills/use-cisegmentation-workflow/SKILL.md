---
name: use-cisegmentation-workflow
description: Configure, launch, monitor, and recover the Bilayers CI Segmentation workflow for OME-Zarr images or HCS plates using Cellpose, Cellpose-SAM, StarDist, InstanSeg, or Spotiflow.
metadata:
  version: "2"
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
6. Obtain explicit confirmation immediately before submission.
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

Require explicit confirmation before submission, cancellation, deletion, or
overwrite. A scheduler completion alone is not proof of successful output
creation.
