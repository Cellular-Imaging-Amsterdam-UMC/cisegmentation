# Troubleshooting

| Failure class | Check | User-actionable next step | Retry safety |
| --- | --- | --- | --- |
| Invalid selection | Selected object type, active group, readable images, and supported image/dataset/plate/screen scope | Select an authorized supported object in the intended group. | Safe before submission. |
| Missing input | Conversion or staging reports no top-level OME-Zarr input | Verify the OMERO object still exists and is readable; restage through the authenticated workflow capability. | Safe only when no run ID was created. |
| Invalid parameter | Descriptor value, enabled step, channel count, range, and conditional rules | Correct the rejected field using `PARAMETERS.md` and the configured descriptor. | Safe before submission. |
| Insufficient permission | Read access to the source and permission to create or attach outputs in the active group | Ask an OMERO administrator or group owner for the required access; do not change groups implicitly. | Retry after permissions change and only if no active run exists. |
| Unavailable resource | Requested CUDA/GPU or workflow queue is unavailable | Use `device=auto` or `cpu` if acceptable, or wait for the configured resource. | Submit once after changing the resolved request. |
| Delayed status or client timeout | Existing run ID and consumer-reported state | Continue monitoring the same run ID. Do not create a replacement solely because a response was delayed. | Do not resubmit while state is unknown or active. |
| Workflow failure | Final run status and sanitized workflow error | Correct a user-controlled input/parameter issue, then request confirmation for a new run. Escalate infrastructure failures without exposing internal details. | New submission requires confirmation. |
| Cancellation | Whether cancellation is supported and the current run state | Show the run ID and side effects, then obtain explicit confirmation before cancelling. | Never cancel twice or assume partial outputs were removed. |
| Missing output | Successful run status plus expected OMERO image/plate and FileAnnotations | Refresh output discovery; compare requested output mode and database setting. Escalate if final artifacts remain absent. | Do not rerun automatically. |
| Partial output | Some fields, labels, or annotations are absent | Preserve available evidence and report which expected artifacts failed validation. | Rerun only after explicit confirmation and duplicate-output review. |

Never expose tokens, usernames, private paths, internal hostnames, scheduler
credentials, or server configuration. Never bypass OMERO permissions or change
the configured workflow repository/ref. Use only consumer-provided typed,
authenticated capabilities.
