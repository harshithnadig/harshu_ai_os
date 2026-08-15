# ==============================================================================
# Harshu AI OS - Run Smoke Tests for OmniRoute Gateway Subsystem
# ==============================================================================

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Running OmniRoute Subsystem Smoke Tests" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$RunnerPath = Join-Path $PSScriptRoot "..\tests\run_all_smoke_tests.py"
uv run python $RunnerPath
