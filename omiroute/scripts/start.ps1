# ==============================================================================
# Harshu AI OS - Start OmniRoute Gateway Subsystem
# ==============================================================================

$Port = 20128
$HostAddr = "127.0.0.1"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Starting OmniRoute Gateway on http://$HostAddr`:$Port" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Check if gateway is already running on the target port
$Existing = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($Existing) {
    Write-Host "[INFO] OmniRoute Gateway is already running on port $Port (PID: $($Existing[0].OwningProcess))." -ForegroundColor Green
    exit 0
}

# Ensure environment variables are loaded if .env exists in omiroute/
$EnvPath = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $EnvPath) {
    Write-Host "[INFO] Loading environment variables from omiroute/.env..." -ForegroundColor Yellow
    Get-Content $EnvPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            $varName = $parts[0].Trim()
            $varVal = $parts[1].Trim()
            [System.Environment]::SetEnvironmentVariable($varName, $varVal, [System.EnvironmentVariableTarget]::Process)
        }
    }
}

$LocalMjs = Join-Path $PSScriptRoot "..\upstream\OmniRoute\bin\omniroute.mjs"

if (Test-Path $LocalMjs) {
    Write-Host "[EXEC] Starting via local upstream distribution..." -ForegroundColor Green
    Start-Process -FilePath "node" -ArgumentList "`"$LocalMjs`" serve --port $Port --no-open" -NoNewWindow
} elseif (Get-Command "omniroute" -ErrorAction SilentlyContinue) {
    Write-Host "[EXEC] Starting via global omniroute binary..." -ForegroundColor Green
    Start-Process -FilePath "omniroute" -ArgumentList "serve --port $Port --no-open" -NoNewWindow
} elseif (Get-Command "npx" -ErrorAction SilentlyContinue) {
    Write-Host "[EXEC] Starting via npx omniroute..." -ForegroundColor Green
    Start-Process -FilePath "npx" -ArgumentList "-y omniroute@3.8.49 serve --port $Port --no-open" -NoNewWindow
} else {
    Write-Error "[ERROR] Neither local omniroute distribution nor node was found. Please install Node.js."
    exit 1
}

Start-Sleep -Seconds 3
$Check = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($Check) {
    Write-Host "[SUCCESS] OmniRoute Gateway is active and listening on http://$HostAddr`:$Port/v1" -ForegroundColor Green
} else {
    Write-Host "[NOTICE] Process started. Gateway may take a few seconds to finish initializing database and listening on port $Port." -ForegroundColor Yellow
}
