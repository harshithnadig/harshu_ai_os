# ==============================================================================
# Harshu AI OS - Local Production Deployment Simulation Script
# ==============================================================================
# Deploys an immutable GHCR Docker image tag to a local container simulation
# with preflight validation, health check polling, and automated rollback.
# ==============================================================================

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0, HelpMessage = "Immutable GHCR image tag to deploy (e.g., sha-3d4f8eb)")]
    [ValidateNotNullOrEmpty()]
    [string]$ImageTag,

    [Parameter(Mandatory = $false)]
    [int]$HealthTimeoutSeconds = 30,

    [Parameter(Mandatory = $false)]
    [int]$HealthIntervalSeconds = 2
)

$ErrorActionPreference = 'Stop'
if (Test-Path variable:global:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$EnvFile = Join-Path $RepoRoot '.env'
$DataDir = Join-Path $RepoRoot 'data'
$ContainerName = 'harshu-ai-os'
$RegistryBase = 'ghcr.io/harshithnadig/harshu_ai_os'

Write-Host ''
Write-Host '==================================================' -ForegroundColor Cyan
Write-Host '  HARSHU AI OS - Local Deployment Simulation' -ForegroundColor Cyan
Write-Host '==================================================' -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# 1. Safety & Tag Validation
# ------------------------------------------------------------------------------
$cleanTag = $ImageTag.Trim()
if ($cleanTag.StartsWith("$RegistryBase`:", [System.StringComparison]::OrdinalIgnoreCase)) {
    $cleanTag = $cleanTag.Substring("$RegistryBase`:".Length)
}

if ([string]::IsNullOrWhiteSpace($cleanTag)) {
    Write-Host '[ERROR] Image tag cannot be empty.' -ForegroundColor Red
    exit 1
}

if ($cleanTag -eq 'latest' -or $cleanTag -like '*:latest') {
    Write-Host '[ERROR] Safety violation: Deploying mutable tag "latest" is prohibited.' -ForegroundColor Red
    Write-Host '        Please provide an immutable tag (e.g. sha-3d4f8eb).' -ForegroundColor Yellow
    exit 1
}

if (-not $cleanTag.StartsWith('sha-', [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "[WARNING] Image tag '$cleanTag' does not follow the immutable 'sha-*' naming convention." -ForegroundColor Yellow
}

$FullImage = "$RegistryBase`:$cleanTag"
Write-Host "[TARGET] Deployment candidate: $FullImage" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# 2. Preflight Checks
# ------------------------------------------------------------------------------
Write-Host '--- Step 1: Preflight Checks ---' -ForegroundColor White

# Check Docker daemon availability
Write-Host 'Checking Docker daemon...' -NoNewline
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker command failed with exit code $LASTEXITCODE"
    }
    Write-Host ' [OK]' -ForegroundColor Green
} catch {
    Write-Host ' [FAILED]' -ForegroundColor Red
    Write-Host '[ERROR] Docker daemon is not running or unreachable.' -ForegroundColor Red
    Write-Host '        Please start Docker Desktop and try again.' -ForegroundColor Yellow
    exit 1
}

# Check .env file existence
Write-Host 'Checking .env configuration...' -NoNewline
if (-not (Test-Path -Path $EnvFile -PathType Leaf)) {
    Write-Host ' [FAILED]' -ForegroundColor Red
    Write-Host "[ERROR] Environment file '$EnvFile' was not found." -ForegroundColor Red
    Write-Host '        Create a valid .env file from .env.example before deploying.' -ForegroundColor Yellow
    exit 1
}
Write-Host ' [OK]' -ForegroundColor Green

# Check data directory existence
Write-Host 'Checking data volume directory...' -NoNewline
if (-not (Test-Path -Path $DataDir -PathType Container)) {
    Write-Host ' [FAILED]' -ForegroundColor Red
    Write-Host "[ERROR] Data directory '$DataDir' was not found." -ForegroundColor Red
    exit 1
}
Write-Host ' [OK]' -ForegroundColor Green

# Pull candidate image from GHCR
Write-Host "Pulling candidate image '$FullImage' from GHCR..." -ForegroundColor Yellow
$pullOutput = docker pull $FullImage 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host '[ERROR] Failed to pull image from GHCR.' -ForegroundColor Red
    Write-Host ($pullOutput | Out-String) -ForegroundColor Red
    exit 1
}
Write-Host 'Image pulled successfully.' -ForegroundColor Green

# Resolve image digest
$imageDigest = (docker image inspect $FullImage --format '{{index .RepoDigests 0}}' 2>$null)
if (-not $imageDigest) {
    $imageDigest = (docker image inspect $FullImage --format '{{.Id}}' 2>$null)
}
Write-Host "[DIGEST] Resolved Image: $imageDigest" -ForegroundColor Gray

# ------------------------------------------------------------------------------
# 3. Capture Current State & Rollback Candidate
# ------------------------------------------------------------------------------
Write-Host ''
Write-Host '--- Step 2: Container State Inspection ---' -ForegroundColor White

$previousImage = $null
$previousContainerRunning = $false

$existingContainerId = (docker ps -a -q --filter "name=^/${ContainerName}$" 2>$null)
if (-not $existingContainerId) {
    $existingContainerId = (docker ps -a -q --filter "name=^${ContainerName}$" 2>$null)
}

if ($existingContainerId) {
    $previousImage = (docker inspect $ContainerName --format '{{.Config.Image}}' 2>$null)
    $previousRunningState = (docker inspect $ContainerName --format '{{.State.Running}}' 2>$null)
    $previousContainerRunning = ($previousRunningState -eq 'true')
    
    Write-Host "[STATE] Found existing container '$ContainerName'." -ForegroundColor Yellow
    Write-Host "        Previous Image: $previousImage (Running: $previousContainerRunning)" -ForegroundColor Yellow
    Write-Host "        Stopping and removing previous container..." -ForegroundColor Yellow
    
    docker stop $ContainerName 2>&1 | Out-Null
    docker rm $ContainerName 2>&1 | Out-Null
} else {
    Write-Host "[STATE] No existing container named '$ContainerName' found. Clean slate deployment." -ForegroundColor Green
}

# Helper function to launch container
function Start-HarshuContainer {
    param(
        [string]$ImageToRun
    )
    $runResult = docker run -d `
        --name $ContainerName `
        -p 8000:8000 `
        --env-file $EnvFile `
        -v "${DataDir}:/app/data" `
        $ImageToRun 2>&1
    
    return $runResult
}

# Helper function to perform health check
function Test-ContainerHealth {
    param(
        [int]$TimeoutSeconds,
        [int]$IntervalSeconds
    )
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $healthUrl = 'http://127.0.0.1:8000/health'
    
    Write-Host "Polling health endpoint ($healthUrl)..." -NoNewline
    
    while ($stopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        # Check if container died prematurely
        $isRunning = docker inspect $ContainerName --format '{{.State.Running}}' 2>$null
        if ($isRunning -ne 'true') {
            Write-Host ' [CONTAINER CRASHED]' -ForegroundColor Red
            return @{ Success = $false; Reason = 'Container stopped unexpectedly during startup' }
        }
        
        try {
            $response = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3 -ErrorAction Stop
            if ($response -and $response.status -eq 'healthy') {
                $secondsTaken = [math]::Round($stopwatch.Elapsed.TotalSeconds, 1)
                Write-Host " [HEALTHY in ${secondsTaken}s]" -ForegroundColor Green
                return @{ Success = $true; Response = $response; Elapsed = $stopwatch.Elapsed.TotalSeconds }
            }
        } catch {
            # Service still booting, continue polling
        }
        
        Write-Host '.' -NoNewline
        Start-Sleep -Seconds $IntervalSeconds
    }
    
    Write-Host ' [TIMEOUT]' -ForegroundColor Red
    return @{ Success = $false; Reason = "Health endpoint did not return healthy within $TimeoutSeconds seconds" }
}

# ------------------------------------------------------------------------------
# 4. Deploy Candidate Container
# ------------------------------------------------------------------------------
Write-Host ''
Write-Host '--- Step 3: Deploying Candidate ---' -ForegroundColor White
Write-Host "Launching container '$ContainerName' with $FullImage..." -ForegroundColor Cyan

$candidateRunId = Start-HarshuContainer -ImageToRun $FullImage

if ($LASTEXITCODE -ne 0 -or -not $candidateRunId) {
    Write-Host "[ERROR] Failed to start container with candidate image." -ForegroundColor Red
    Write-Host ($candidateRunId | Out-String) -ForegroundColor Red
    $deploymentHealthy = @{ Success = $false; Reason = 'docker run command failed' }
} else {
    $shortCandidateId = $candidateRunId.Trim()
    Write-Host "Container started with ID: $shortCandidateId" -ForegroundColor Green
    $deploymentHealthy = Test-ContainerHealth -TimeoutSeconds $HealthTimeoutSeconds -IntervalSeconds $HealthIntervalSeconds
}

# ------------------------------------------------------------------------------
# 5. Handle Rollback if Verification Failed
# ------------------------------------------------------------------------------
if (-not $deploymentHealthy.Success) {
    Write-Host ''
    Write-Host '==================================================' -ForegroundColor Red
    Write-Host '  DEPLOYMENT FAILED - INITIATING ROLLBACK' -ForegroundColor Red
    Write-Host '==================================================' -ForegroundColor Red
    Write-Host "Reason: $($deploymentHealthy.Reason)" -ForegroundColor Yellow
    
    # Collect logs from failed candidate
    Write-Host '--- Failed Candidate Logs ---' -ForegroundColor Yellow
    docker logs --tail 30 $ContainerName 2>&1 | Write-Host -ForegroundColor DarkGray
    
    # Stop and clean failed container
    Write-Host "Stopping and cleaning up failed container '$ContainerName'..." -ForegroundColor Yellow
    docker stop $ContainerName 2>&1 | Out-Null
    docker rm $ContainerName 2>&1 | Out-Null
    
    if ($previousImage) {
        Write-Host "Attempting rollback to previous image: $previousImage" -ForegroundColor Cyan
        $rollbackRunId = Start-HarshuContainer -ImageToRun $previousImage
        
        if ($LASTEXITCODE -eq 0 -and $rollbackRunId) {
            $rollbackHealth = Test-ContainerHealth -TimeoutSeconds $HealthTimeoutSeconds -IntervalSeconds $HealthIntervalSeconds
            if ($rollbackHealth.Success) {
                Write-Host ''
                Write-Host '==================================================' -ForegroundColor Yellow
                Write-Host '  ROLLBACK SUCCESSFUL' -ForegroundColor Yellow
                Write-Host '==================================================' -ForegroundColor Yellow
                Write-Host "Restored previous deployment: $previousImage" -ForegroundColor Green
                Write-Host "Container state: Running & Healthy" -ForegroundColor Green
                exit 1
            } else {
                Write-Host '[CRITICAL] Rollback container started but failed health check.' -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "[CRITICAL] Failed to restart previous image '$previousImage'." -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host '[INFO] No previous deployment existed to roll back to. Clean failure.' -ForegroundColor Yellow
        exit 1
    }
}

# ------------------------------------------------------------------------------
# 6. Structured Deployment Summary
# ------------------------------------------------------------------------------
$finalContainerId = (docker inspect $ContainerName --format '{{.Id}}' 2>$null)
$finalImageRunning = (docker inspect $ContainerName --format '{{.Config.Image}}' 2>$null)
$finalStatus = (docker inspect $ContainerName --format '{{.State.Status}}' 2>$null)

$displayContainerId = if ($finalContainerId) { $finalContainerId.Substring(0, [math]::Min(12, $finalContainerId.Length)) } else { "N/A" }
$displayPrevImage = if ($previousImage) { $previousImage } else { "None (Initial deploy)" }
$displayElapsed = [math]::Round($deploymentHealthy.Elapsed, 2)

Write-Host ''
Write-Host '==================================================' -ForegroundColor Green
Write-Host '  DEPLOYMENT SUCCESSFUL' -ForegroundColor Green
Write-Host '==================================================' -ForegroundColor Green
Write-Host "Target Image:      $FullImage" -ForegroundColor Green
Write-Host "Image Digest:      $imageDigest" -ForegroundColor Gray
Write-Host "Previous Image:    $displayPrevImage" -ForegroundColor Gray
Write-Host "Container Name:    $ContainerName" -ForegroundColor White
Write-Host "Container ID:      $displayContainerId" -ForegroundColor White
Write-Host "Container Status:  $finalStatus" -ForegroundColor Green
Write-Host "Health Check:      HTTP 200 OK - status: healthy (${displayElapsed}s)" -ForegroundColor Green
Write-Host "API Endpoint:      http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host '==================================================' -ForegroundColor Green
Write-Host ''
