# ==============================================================================
# Harshu AI OS - Unified Local Development Launcher
# ==============================================================================

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RunDir = Join-Path $RepoRoot '.run'
$PidFile = Join-Path $RunDir 'dev-pids.json'

if (-not (Test-Path $RunDir)) {
    New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
}

Write-Host '==================================================' -ForegroundColor Cyan
Write-Host '  HARSHU AI OS — Local Dev Launcher' -ForegroundColor Cyan
Write-Host '==================================================' -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# 1. Dependency Checks
# ------------------------------------------------------------------------------
$MissingDeps = [System.Collections.Generic.List[string]]::new()

if (-not (Get-Command 'node' -ErrorAction SilentlyContinue)) {
    $MissingDeps.Add('Node.js (node)')
}
if (-not (Get-Command 'npm' -ErrorAction SilentlyContinue)) {
    $MissingDeps.Add('npm')
}
if (-not (Get-Command 'uv' -ErrorAction SilentlyContinue)) {
    $MissingDeps.Add('uv (Python package manager)')
}

if ($MissingDeps.Count -gt 0) {
    Write-Host '[ERROR] Missing required dependencies:' -ForegroundColor Red
    foreach ($dep in $MissingDeps) {
        Write-Host "  - $dep" -ForegroundColor Red
    }
    Write-Host 'Please install the missing tools and try again.' -ForegroundColor Yellow
    exit 1
}

# ------------------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------------------
function Get-PortProcessId {
    param([int]$Port)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($conns) {
        return $conns[0].OwningProcess
    }
    return $null
}

function Wait-ForPort {
    param(
        [int]$Port,
        [int]$TimeoutSeconds = 25
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $pidFound = Get-PortProcessId -Port $Port
        if ($pidFound) {
            return $pidFound
        }
        Start-Sleep -Milliseconds 500
    }
    return $null
}

$TrackedPids = @{}
if (Test-Path $PidFile) {
    try {
        $raw = Get-Content $PidFile -Raw | ConvertFrom-Json
        if ($raw) {
            foreach ($prop in $raw.PSObject.Properties) {
                $TrackedPids[$prop.Name] = $prop.Value
            }
        }
    } catch {}
}

# ------------------------------------------------------------------------------
# 3. Service 1: OmniRoute Gateway (Port 20128)
# ------------------------------------------------------------------------------
$OmniPort = 20128
$OmniPid = Get-PortProcessId -Port $OmniPort

if ($OmniPid) {
    Write-Host "[INFO] OmniRoute is already active on port $OmniPort (PID: $OmniPid)." -ForegroundColor Green
    $TrackedPids['omniroute'] = $OmniPid
} else {
    Write-Host '[START] Launching OmniRoute Gateway subsystem...' -ForegroundColor Yellow
    $OmniScript = Join-Path $RepoRoot 'omiroute\scripts\start.ps1'
    
    if (Test-Path $OmniScript) {
        $omniProc = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $OmniScript) `
            -WorkingDirectory (Join-Path $RepoRoot 'omiroute') `
            -WindowStyle Hidden `
            -PassThru
    } else {
        $omniProc = Start-Process -FilePath 'npx.cmd' `
            -ArgumentList @('-y', 'omniroute@3.8.49', 'serve', '--port', "$OmniPort", '--no-open') `
            -WorkingDirectory (Join-Path $RepoRoot 'omiroute') `
            -WindowStyle Hidden `
            -PassThru
    }

    $OmniPid = Wait-ForPort -Port $OmniPort -TimeoutSeconds 25
    if ($OmniPid) {
        $TrackedPids['omniroute'] = $OmniPid
    } elseif ($omniProc -and -not $omniProc.HasExited) {
        $TrackedPids['omniroute'] = $omniProc.Id
    } else {
        Write-Host '[WARNING] OmniRoute did not bind port 20128 within timeout.' -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------------------------
# 4. Service 2: FastAPI Backend (Port 8000)
# ------------------------------------------------------------------------------
$ApiPort = 8000
$ApiPid = Get-PortProcessId -Port $ApiPort

if ($ApiPid) {
    Write-Host "[INFO] FastAPI backend is already active on port $ApiPort (PID: $ApiPid)." -ForegroundColor Green
    $TrackedPids['fastapi'] = $ApiPid
} else {
    Write-Host '[START] Launching FastAPI backend (uv uvicorn)...' -ForegroundColor Yellow
    
    $fastApiProc = Start-Process -FilePath 'uv' `
        -ArgumentList @('run', 'uvicorn', 'harshu_ai_os.api.main:app', '--app-dir', 'src', '--reload', '--port', "$ApiPort") `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -PassThru

    $ApiPid = Wait-ForPort -Port $ApiPort -TimeoutSeconds 25
    if ($ApiPid) {
        $TrackedPids['fastapi'] = $ApiPid
    } elseif ($fastApiProc -and -not $fastApiProc.HasExited) {
        $TrackedPids['fastapi'] = $fastApiProc.Id
    } else {
        Write-Host '[WARNING] FastAPI did not bind port 8000 within timeout.' -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------------------------
# 5. Service 3: Frontend (Port 5173)
# ------------------------------------------------------------------------------
$FrontendPort = 5173
$FrontendPid = Get-PortProcessId -Port $FrontendPort

if ($FrontendPid) {
    Write-Host "[INFO] Frontend dev server is already active on port $FrontendPort (PID: $FrontendPid)." -ForegroundColor Green
    $TrackedPids['frontend'] = $FrontendPid
} else {
    Write-Host '[START] Launching Vite frontend (npm run dev)...' -ForegroundColor Yellow
    $FrontendDir = Join-Path $RepoRoot 'frontend'
    
    $frontProc = Start-Process -FilePath 'npm.cmd' `
        -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', "$FrontendPort") `
        -WorkingDirectory $FrontendDir `
        -WindowStyle Hidden `
        -PassThru

    $FrontendPid = Wait-ForPort -Port $FrontendPort -TimeoutSeconds 20
    if ($FrontendPid) {
        $TrackedPids['frontend'] = $FrontendPid
    } elseif ($frontProc -and -not $frontProc.HasExited) {
        $TrackedPids['frontend'] = $frontProc.Id
    } else {
        Write-Host '[WARNING] Frontend dev server did not bind port 5173 within timeout.' -ForegroundColor Yellow
    }
}

# Save tracked PIDs
$TrackedPids | ConvertTo-Json | Set-Content -Path $PidFile -Force

# ------------------------------------------------------------------------------
# 6. Status Summary
# ------------------------------------------------------------------------------
$omniListening = Get-PortProcessId -Port $OmniPort
$apiListening  = Get-PortProcessId -Port $ApiPort
$frontListening = Get-PortProcessId -Port $FrontendPort

$omniStatus  = if ($omniListening)  { '[READY]' } else { '[PENDING]' }
$apiStatus   = if ($apiListening)   { '[READY]' } else { '[PENDING]' }
$frontStatus = if ($frontListening) { '[READY]' } else { '[PENDING]' }

$omniColor  = if ($omniListening)  { 'Green' } else { 'Yellow' }
$apiColor   = if ($apiListening)   { 'Green' } else { 'Yellow' }
$frontColor = if ($frontListening) { 'Green' } else { 'Yellow' }

Write-Host ''
Write-Host '==================================================' -ForegroundColor Green
Write-Host 'HARSHU AI OS' -ForegroundColor Green
Write-Host '==================================================' -ForegroundColor Green
Write-Host "OmniRoute  $omniStatus http://127.0.0.1:$OmniPort" -ForegroundColor $omniColor
Write-Host "FastAPI    $apiStatus http://127.0.0.1:$ApiPort" -ForegroundColor $apiColor
Write-Host "Frontend   $frontStatus http://127.0.0.1:$FrontendPort" -ForegroundColor $frontColor
Write-Host '==================================================' -ForegroundColor Green
Write-Host 'To stop all services, run: .\scripts\stop-dev.ps1' -ForegroundColor Gray
Write-Host ''
