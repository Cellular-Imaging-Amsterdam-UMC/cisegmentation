---
name: use-cisegmentation-workflow
description: Configure, launch, monitor, and explain the CI Segmentation BIOMERO workflow. Use when a user explicitly wants to segment an authorized OMERO image, dataset, plate, or screen with Cellpose, Cellpose-SAM, StarDist, InstanSeg, or Spotiflow.
metadata:
  version: "1"
  biomero-purpose: "workflow-operation"
  biomero-consumers: "omero-biomero"
  biomero-auto-activate: "false"
---

# Use CI Segmentation Workflow

1. Confirm that the user explicitly asked to run CI Segmentation. Selecting or
   viewing an OMERO object is not a run request.
2. Inspect the active OMERO user, group, selected object, configured workflow
   revision, and available workflow capabilities. Preserve the active group and
   permissions. Never invent object IDs or tool capabilities.
3. Read [PARAMETERS.md](references/PARAMETERS.md) before proposing settings.
4. Validate the selected object type, readable inputs, one-based channel
   numbers, enabled segmentation steps, parameter types and ranges, compatible
   model options, output mode, and requested compute device.
5. Present the resolved workflow revision, input object, parameters, expected
   outputs, resource choice, and side effects.
6. Obtain explicit confirmation immediately before submission.
7. Submit exactly once through the consumer's typed, authenticated workflow
   capability. Retain the returned run or job ID.
8. Monitor that ID. Do not resubmit merely because status is delayed or a
   client response times out.
9. Read [OUTPUTS.md](references/OUTPUTS.md) to verify completion and explain
   results. Read [TROUBLESHOOTING.md](references/TROUBLESHOOTING.md) only after
   validation, execution, or output verification fails.
10. Record the workflow key, configured ref, resolved commit, parameters, input
    object, run ID, timestamps, final status, and discovered output objects as
    provenance.

Require explicit confirmation before submission, cancellation, deletion,
overwrite, or attachment changes. Never modify BIOMERO configuration, release
pins, server settings, or permissions. A scheduler completion alone is not
proof of successful output creation.
