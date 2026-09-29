<#
Submit/watch isolated real-checkpoint tests in the existing local Slurm image.
No image is built. Default: all checkpoints, all targets, small 2D/3D checks,
then 8192-pixel streamed regions. Results and logs are collected on V:.
#>
[CmdletBinding()]
param(
    [string]$Repository = 'U:\Ron\Documents\Github\cisegmentation',
    [string]$ResultsRoot = 'V:\BIOMERO-local\tests\results\all-models',
    [string]$ClusterInput = '/data/tilescan-tests/input/1_B02__cells-tilescan-40000x40000.ome.zarr',
    [string]$WorkflowImage,
    [ValidateRange(512, 40000)][int]$LargeSize = 8192,
    [ValidateRange(1, 1440)][int]$ModelTimeoutMinutes = 30,
    [ValidateRange(1, 168)][int]$WallHours = 48,
    [ValidateRange(512, 24564)][int]$GpuMemoryMB = 12288,
    [string[]]$Models,
    [switch]$QuickOnly,
    [switch]$Full40K,
    [switch]$PlanOnly,
    [switch]$SubmitOnly,
    [switch]$RetryFailures,
    [string]$ResumeRun,
    [string]$Scheduler = 'nl-biomero-local-slurm-gpu-slurmctld-1',
    [string]$GpuNode = 'nl-biomero-local-slurm-gpu-c1-1'
)
$ErrorActionPreference = 'Stop'

function Docker-Text {
    param([string[]]$DockerArguments)
    $lines = & docker @DockerArguments
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed: docker $($DockerArguments -join ' ')" }
    return (($lines | Out-String).Trim())
}
function Write-Utf8 {
    param([string]$Path, [string]$Value)
    [IO.File]::WriteAllText($Path, $Value.Replace("`r`n", "`n") + "`n", [Text.UTF8Encoding]::new($false))
}
function Copy-Docker {
    param([string]$From, [string]$To)
    $null = Docker-Text -DockerArguments @('cp', $From, $To)
}

if ($ResumeRun) {
    $RunDirectory = (Resolve-Path -LiteralPath $ResumeRun).Path
    $Metadata = Get-Content -LiteralPath (Join-Path $RunDirectory 'launcher.json') -Raw | ConvertFrom-Json
    $RunId = $Metadata.run_id
    $Scheduler = $Metadata.scheduler
    $GpuNode = $Metadata.gpu_node
    $Image = $Metadata.image
    $ClusterRoot = $Metadata.cluster_root
    $JobId = $Metadata.job_id
    $AlreadyRunning = $false
    if ($JobId) {
        $AlreadyRunning = [bool](Docker-Text -DockerArguments @('exec', $Scheduler, 'squeue', '-h', '-j', "$JobId", '-o', '%T'))
    }
} else {
    if ($Full40K -and $QuickOnly) { throw 'Choose Full40K or QuickOnly, not both.' }
    $Repository = (Resolve-Path -LiteralPath $Repository).Path
    $Version = (Get-Content -LiteralPath (Join-Path $Repository 'version.txt') -Raw).Trim()
    if ($Version -notmatch '^v?\d+\.\d+\.\d+(?:-[\w.-]+)?$') { throw "Invalid repository version: $Version" }
    $Image = $WorkflowImage
    if (-not $Image) { $Image = "/data/my-scratch/singularity_images/workflows/cisegmentation/w_cisegmentation_${Version}.sif" }
    $null = Docker-Text -DockerArguments @('exec', $GpuNode, 'test', '-f', $Image)
    foreach ($name in @('cisegmentation\streaming.py', 'tools\test_model_matrix.py', 'tools\tilescan_resource_smoke.py', 'tests\data\nuclei-spots-cytoplasm.ome.zarr')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Repository $name))) { throw "Missing source/fixture: $name" }
    }
    if (-not $QuickOnly) { $null = Docker-Text -DockerArguments @('exec', $GpuNode, 'test', '-d', $ClusterInput) }
    if ($Full40K) {
        $LargeSize = 40000
        if (-not $PSBoundParameters.ContainsKey('ModelTimeoutMinutes')) { $ModelTimeoutMinutes = 240 }
        if (-not $PSBoundParameters.ContainsKey('WallHours')) { $WallHours = 72 }
    }
    $RunId = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
    $RunDirectory = Join-Path $ResultsRoot $RunId
    $Snapshot = Join-Path $RunDirectory 'code'
    $null = New-Item -ItemType Directory -Path (Join-Path $Snapshot 'tools') -Force
    $null = New-Item -ItemType Directory -Path (Join-Path $Snapshot 'tests\data') -Force
    $null = New-Item -ItemType Directory -Path (Join-Path $RunDirectory 'results') -Force
    Copy-Item -LiteralPath (Join-Path $Repository 'cisegmentation') -Destination $Snapshot -Recurse
    Copy-Item -LiteralPath (Join-Path $Repository 'tests\data\nuclei-spots-cytoplasm.ome.zarr') -Destination (Join-Path $Snapshot 'tests\data') -Recurse
    foreach ($name in @('test_model_matrix.py', 'tilescan_resource_smoke.py', 'test_all_models.ps1')) {
        Copy-Item -LiteralPath (Join-Path $Repository "tools\$name") -Destination (Join-Path $Snapshot 'tools')
    }
    foreach ($name in @('config.yaml', 'pyproject.toml', 'version.txt', 'wrapper.py', 'bilayers_cli.py')) {
        if (Test-Path -LiteralPath (Join-Path $Repository $name)) { Copy-Item -LiteralPath (Join-Path $Repository $name) -Destination $Snapshot }
    }
    $ClusterRoot = "/data/tilescan-tests/all-models/$RunId"
    $null = Docker-Text -DockerArguments @('exec', $GpuNode, 'mkdir', '-p', "$ClusterRoot/code", "$ClusterRoot/results")
    Copy-Docker -From "$Snapshot/." -To "${GpuNode}:$ClusterRoot/code"
    $Config = [ordered]@{
        output = '/testrun/results'; fixture = '/app/tests/data/nuclei-spots-cytoplasm.ome.zarr'
        input = $ClusterInput; large_size = $LargeSize; quick_only = [bool]$QuickOnly
        native_3d = $true; gpu_memory_mb = $GpuMemoryMB; timeout_minutes = $ModelTimeoutMinutes
        models = $null
    }
    if ($Models) { $Config.models = @($Models) }
    Write-Utf8 -Path (Join-Path $RunDirectory 'config.json') -Value ($Config | ConvertTo-Json -Depth 10)
    Copy-Docker -From (Join-Path $RunDirectory 'config.json') -To "${GpuNode}:$ClusterRoot/config.json"
    $Metadata = [pscustomobject]@{
        run_id = $RunId; cluster_root = $ClusterRoot; scheduler = $Scheduler; gpu_node = $GpuNode
        image = $Image; repository = $Repository; job_id = $null; wall_hours = $WallHours
    }
    Write-Utf8 -Path (Join-Path $RunDirectory 'launcher.json') -Value ($Metadata | ConvertTo-Json)
    $AlreadyRunning = $false
}
$null = Docker-Text -DockerArguments @('exec', $GpuNode, 'test', '-f', $Image)
if ($RunId -notmatch '^\d{8}-\d{6}-[a-f0-9]{8}$' -or $ClusterRoot -ne "/data/tilescan-tests/all-models/$RunId") {
    throw 'Invalid saved run identity; refusing an ambiguous Docker path.'
}

$BindArguments = @('--bind', "${ClusterRoot}:/testrun", '--bind', "${ClusterRoot}/code:/app", '--bind', '/data/tilescan-tests/input:/data/tilescan-tests/input:ro')
$null = Docker-Text -DockerArguments (@('exec', $GpuNode, 'singularity', 'exec') + $BindArguments + @($Image, 'python', '/app/tools/test_model_matrix.py', '--config', '/testrun/config.json', '--plan-only'))
Copy-Docker -From "${GpuNode}:$ClusterRoot/results/plan.json" -To (Join-Path $RunDirectory 'results\plan.json')
$Plan = Get-Content -LiteralPath (Join-Path $RunDirectory 'results\plan.json') -Raw | ConvertFrom-Json
Write-Host "Run folder: $RunDirectory"
Write-Host "Plan: $($Plan.models.Count) real checkpoints; $($Plan.total_cases) model/target/stage cases."
if ($Plan.checkpoint_inventory) {
    $CachedCount = @($Plan.checkpoint_inventory | Where-Object cached).Count
    Write-Host "Cached checkpoints: $CachedCount/$($Plan.models.Count)"
    if ($Plan.missing_checkpoints) { Write-Warning "Missing cached weights: $($Plan.missing_checkpoints -join ', ')" }
}
if ($PlanOnly) { Write-Host 'Plan saved. No Slurm job submitted.'; return }

$AlreadyFinished = $false
if ($ResumeRun -and -not $AlreadyRunning) {
    $null = & docker exec $GpuNode test -f "$ClusterRoot/results/summary.json"
    if ($LASTEXITCODE -eq 0) {
        Copy-Docker -From "${GpuNode}:$ClusterRoot/results/summary.json" -To (Join-Path $RunDirectory 'results\summary.json')
        $PreviousSummary = Get-Content -LiteralPath (Join-Path $RunDirectory 'results\summary.json') -Raw | ConvertFrom-Json
        $AlreadyFinished = $PreviousSummary.status -in @('completed', 'completed_with_failures')
    }
}
if (-not $AlreadyRunning -and (-not $AlreadyFinished -or $RetryFailures)) {
    $RetryFlag = $(if ($RetryFailures) { '--retry-failures' } else { '' })
    $TimeLimit = ([TimeSpan]::FromHours([int]$Metadata.wall_hours)).ToString('d\-hh\:mm\:ss')
    # Only the validated generated run ID is interpolated into shell paths.
    $Batch = @"
#!/bin/bash
#SBATCH --job-name=ciseg-all-models
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclusive
#SBATCH --time=$TimeLimit
#SBATCH --output=$ClusterRoot/coordinator.log
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
singularity exec --nv --bind ${ClusterRoot}:/testrun --bind ${ClusterRoot}/code:/app --bind /data/tilescan-tests/input:/data/tilescan-tests/input:ro $Image python /app/tools/test_model_matrix.py --config /testrun/config.json $RetryFlag
"@
    Write-Utf8 -Path (Join-Path $RunDirectory 'run.sbatch') -Value $Batch
    Copy-Docker -From (Join-Path $RunDirectory 'run.sbatch') -To "${Scheduler}:$ClusterRoot/run.sbatch"
    $Submit = Docker-Text -DockerArguments @('exec', $Scheduler, 'sbatch', '--parsable', "$ClusterRoot/run.sbatch")
    $JobId = ($Submit -split ';')[0].Trim()
    if ($JobId -notmatch '^\d+$') { throw "Unexpected Slurm submission response: $Submit" }
    $Metadata.job_id = $JobId
    Write-Utf8 -Path (Join-Path $RunDirectory 'launcher.json') -Value ($Metadata | ConvertTo-Json)
}
Write-Host "Slurm job: $JobId. To cancel: docker exec $Scheduler scancel $JobId"
if ($SubmitOnly) { Write-Host 'Job submitted. Use -ResumeRun with this run folder to watch/collect later.'; return }
Write-Host 'Watching. Ctrl+C stops this watcher; the Slurm job continues. ResumeRun reconnects.'
$PreviousProgress = ''
while ($true) {
    $State = Docker-Text -DockerArguments @('exec', $Scheduler, 'squeue', '-h', '-j', "$JobId", '-o', '%T')
    if (-not $State) { break }
    $HasSummary = & docker exec $GpuNode test -f "$ClusterRoot/results/summary.json"
    if ($LASTEXITCODE -eq 0) {
        Copy-Docker -From "${GpuNode}:$ClusterRoot/results/summary.json" -To (Join-Path $RunDirectory 'results\summary.json')
        $Summary = Get-Content -LiteralPath (Join-Path $RunDirectory 'results\summary.json') -Raw | ConvertFrom-Json
        $Progress = "$State - $($Summary.completed_cases)/$($Summary.total_cases) complete; $($Summary.execution_counts | ConvertTo-Json -Compress)"
        $null = & docker exec $GpuNode test -f "$ClusterRoot/results/current-case.json"
        if ($LASTEXITCODE -eq 0) {
            Copy-Docker -From "${GpuNode}:$ClusterRoot/results/current-case.json" -To (Join-Path $RunDirectory 'results\current-case.json')
            $CurrentCase = Get-Content -LiteralPath (Join-Path $RunDirectory 'results\current-case.json') -Raw | ConvertFrom-Json
            $Progress += " | $($CurrentCase.model) $($CurrentCase.target) $($CurrentCase.mode) $($CurrentCase.stage)"
            Write-Progress -Activity 'Real checkpoint tests' -Status $Progress -PercentComplete (100 * $Summary.completed_cases / $Summary.total_cases)
        }
    } else { $Progress = $State }
    if ($Progress -ne $PreviousProgress) { Write-Host "$(Get-Date -Format 'HH:mm:ss') $Progress"; $PreviousProgress = $Progress }
    Start-Sleep -Seconds 20
}
Write-Progress -Activity 'Real checkpoint tests' -Completed
Write-Host 'Collecting reports, logs and small comparison arrays. Large label stores stay in Docker.'
$null = Docker-Text -DockerArguments (@('exec', $GpuNode, 'singularity', 'exec') + $BindArguments + @($Image, 'python', '/app/tools/test_model_matrix.py', '--config', '/testrun/config.json', '--collect-only'))
Copy-Docker -From "${GpuNode}:$ClusterRoot/results/evidence.tar.gz" -To (Join-Path $RunDirectory 'evidence.tar.gz')
Copy-Docker -From "${Scheduler}:$ClusterRoot/coordinator.log" -To (Join-Path $RunDirectory 'coordinator.log')
& tar -xzf (Join-Path $RunDirectory 'evidence.tar.gz') -C (Join-Path $RunDirectory 'results')
if ($LASTEXITCODE -ne 0) { throw 'Evidence extraction failed; the archive was preserved.' }
Write-Host "Results ready: $RunDirectory\results\summary.csv"
Write-Host "Large overlays: $GpuNode`:$ClusterRoot/results/cases"
if (Test-Path -LiteralPath (Join-Path $RunDirectory 'results\summary.json')) {
    $Summary = Get-Content -LiteralPath (Join-Path $RunDirectory 'results\summary.json') -Raw | ConvertFrom-Json
    Write-Host "Outcome: $($Summary.status) - $($Summary.execution_counts | ConvertTo-Json -Compress)"
} else { Write-Warning 'Coordinator ended before producing a summary. Examine coordinator.log.' }
