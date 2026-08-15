# ==============================================================================
# Harshu AI OS - Stop OmniRoute Gateway Subsystem
# ==============================================================================

$Port = 20128
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Stopping OmniRoute Gateway on port $Port" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$Connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue

if (-not $Connections) {
    Write-Host "[INFO] No active OmniRoute Gateway process found on port $Port." -ForegroundColor Yellow
    exit 0
}

$Processes = $Connections | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($PidNum in $Processes) {
    try {
        $Proc = Get-Process -Id $PidNum -ErrorAction Stop
        Write-Host "[EXEC] Terminating OmniRoute process '$($Proc.ProcessName)' (PID: $PidNum)..." -ForegroundColor Yellow
        Stop-Process -Id $PidNum -Force
        Write-Host "[SUCCESS] OmniRoute process $PidNum stopped." -ForegroundColor Green
    } catch {
        Write-Warning "[WARN] Could not terminate process $PidNum: $_"
    }
}
