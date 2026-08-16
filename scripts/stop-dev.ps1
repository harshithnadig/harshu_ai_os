# ==============================================================================
# Harshu AI OS - Unified Local Development Shutdown
# ==============================================================================

[CmdletBinding()]
param()

$ErrorActionPreference = 'SilentlyContinue'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RunDir = Join-Path $RepoRoot '.run'
$PidFile = Join-Path $RunDir 'dev-pids.json'

Write-Host '==================================================' -ForegroundColor Cyan
Write-Host '  Stopping Harshu AI OS Development Stack...' -ForegroundColor Cyan
Write-Host '==================================================' -ForegroundColor Cyan

$StoppedAny = $false
$TargetPorts = [ordered]@{
    'OmniRoute' = 20128
    'FastAPI'   = 8000
    'Frontend'  = 5173
}

$KilledPids = [System.Collections.Generic.HashSet[int]]::new()

# 1. Stop Tracked PIDs from state file
if (Test-Path $PidFile) {
    try {
        $raw = Get-Content $PidFile -Raw | ConvertFrom-Json
        if ($raw) {
            foreach ($prop in $raw.PSObject.Properties) {
                $serviceName = $prop.Name
                $procId = [int]$prop.Value
                if ($procId -gt 0 -and (-not $KilledPids.Contains($procId))) {
                    try {
                        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
                        if ($proc) {
                            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
                            $KilledPids.Add($procId) | Out-Null
                            Write-Host ("{0,-10} [STOPPED] (PID: {1})" -f $serviceName, $procId) -ForegroundColor Yellow
                            $StoppedAny = $true
                        }
                    } catch {}
                }
            }
        }
    } catch {}
    Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
}

# 2. Check and stop any remaining processes listening on target ports (scoped cleanup)
foreach ($service in $TargetPorts.Keys) {
    $port = $TargetPorts[$service]
    try {
        $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($conns) {
            foreach ($c in $conns) {
                $pId = [int]$c.OwningProcess
                if ($pId -gt 0 -and (-not $KilledPids.Contains($pId))) {
                    try {
                        Stop-Process -Id $pId -Force -ErrorAction SilentlyContinue
                        $KilledPids.Add($pId) | Out-Null
                        Write-Host ("{0,-10} [STOPPED] (Port: {1}, PID: {2})" -f $service, $port, $pId) -ForegroundColor Yellow
                        $StoppedAny = $true
                    } catch {}
                }
            }
        } else {
            Write-Host ("{0,-10} [INACTIVE] (Port: {1})" -f $service, $port) -ForegroundColor DarkGray
        }
    } catch {
        Write-Host ("{0,-10} [INACTIVE] (Port: {1})" -f $service, $port) -ForegroundColor DarkGray
    }
}

Write-Host '==================================================' -ForegroundColor Cyan
if ($StoppedAny) {
    Write-Host 'All development services safely stopped.' -ForegroundColor Green
} else {
    Write-Host 'No active development services were running.' -ForegroundColor DarkGray
}
Write-Host ''
exit 0
