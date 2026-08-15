# ==============================================================================
# Harshu AI OS - Health Check for OmniRoute Gateway Subsystem
# ==============================================================================

$Endpoint = "http://127.0.0.1:20128/v1/models"
$MonitoringUrl = "http://127.0.0.1:20128/api/monitoring/health"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Checking OmniRoute Gateway Health ($Endpoint)" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Uri $Endpoint -Method Get -TimeoutSec 5 -ErrorAction Stop
    $sw.Stop()

    Write-Host "[STATUS]  OK (HTTP 200)" -ForegroundColor Green
    Write-Host "[LATENCY] $($sw.ElapsedMilliseconds) ms" -ForegroundColor Green
    
    if ($response.data) {
        $count = $response.data.Count
        Write-Host "[MODELS]  $count models available through gateway" -ForegroundColor Green
    }
} catch {
    Write-Host "[STATUS]  UNREACHABLE / ERROR" -ForegroundColor Red
    Write-Host "[DETAIL]  $_" -ForegroundColor Yellow
    Write-Host "`nTo start the gateway, run:`n  powershell -File omiroute/scripts/start.ps1" -ForegroundColor Cyan
    exit 1
}
