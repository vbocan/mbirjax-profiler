#Requires -Version 5.1
<#
.SYNOPSIS
    MBIRJAX GPU Profiler — XLA-level profiling for FPGA candidate discovery

.EXAMPLE
    .\start.ps1
#>

$ErrorActionPreference = "Stop"

# Validate Docker
try {
    $null = docker info 2>&1 | Out-Null
} catch {
    Write-Host ""
    Write-Host "Error: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

function Show-Banner {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  MBIRJAX GPU Profiler (XLA / TensorBoard)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Menu {
    Write-Host "  [R] Run profiler  (captures XLA traces + cost analysis + HLO)" -ForegroundColor White
    Write-Host "  [T] TensorBoard   (view XLA traces in browser)" -ForegroundColor White
    Write-Host "  [Q] Quit" -ForegroundColor White
    Write-Host ""
}

function Run-Profile {
    Write-Host ""
    Write-Host "Running GPU profiler..." -ForegroundColor Cyan
    Write-Host "Volume sizes: 32, 64, 128, 256" -ForegroundColor Gray
    Write-Host "Runs per size: 3 (run 2 is traced)" -ForegroundColor Gray
    Write-Host ""

    Set-Location $PSScriptRoot

    docker compose run --rm mbirjax-profiler python /scripts/comprehensive_profiler.py

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[FAIL] Profiling failed" -ForegroundColor Red
        Read-Host "Press Enter to continue"
        return
    }

    Write-Host ""
    Write-Host "[OK] Profiling completed" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output:" -ForegroundColor Cyan
    Write-Host "  Timing + cost:  output/mbirjax_profile_*.json" -ForegroundColor Gray
    Write-Host "  XLA traces:     output/jax_traces/" -ForegroundColor Gray
    Write-Host "  HLO text:       output/hlo_dumps/" -ForegroundColor Gray
    Write-Host "  HLO graphs:     output/hlo_dumps_xla/*.html  (open in browser)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Next: press [T] to launch TensorBoard" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Start-TensorBoard {
    # Find the most recent trace directory
    $traceDirs = Get-ChildItem -Path "output/jax_traces" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending

    if ($traceDirs.Count -eq 0) {
        Write-Host ""
        Write-Host "No XLA traces found. Run the profiler first." -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    $latest = $traceDirs[0].Name

    Write-Host ""
    Write-Host "Available trace sessions:" -ForegroundColor Cyan
    Write-Host ""

    for ($i = 0; $i -lt $traceDirs.Count; $i++) {
        $dir = $traceDirs[$i]
        $volDirs = (Get-ChildItem -Path $dir.FullName -Directory -ErrorAction SilentlyContinue).Count
        Write-Host "  [$($i + 1)] $($dir.Name)  ($volDirs volume sizes)" -ForegroundColor White
    }

    Write-Host ""
    $selection = Read-Host "Select session (Enter for latest, 'q' to cancel)"

    if ($selection -eq 'q') { return }

    if ([string]::IsNullOrWhiteSpace($selection)) {
        $selected = $latest
    } elseif ([int]::TryParse($selection, [ref]$null) -and [int]$selection -ge 1 -and [int]$selection -le $traceDirs.Count) {
        $selected = $traceDirs[[int]$selection - 1].Name
    } else {
        Write-Host "Invalid selection" -ForegroundColor Red
        Read-Host "Press Enter to continue"
        return
    }

    $logdir = "/output/jax_traces/$selected"

    Write-Host ""
    Write-Host "Launching TensorBoard..." -ForegroundColor Cyan
    Write-Host "  Log dir: $logdir" -ForegroundColor Gray
    Write-Host "  URL:     http://localhost:6006" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press Ctrl+C to stop TensorBoard" -ForegroundColor Yellow
    Write-Host ""

    Set-Location $PSScriptRoot
    docker compose run --rm -p 6006:6006 mbirjax-profiler tensorboard --logdir="$logdir" --host=0.0.0.0 --port=6006
}

# Main loop
Set-Location $PSScriptRoot

while ($true) {
    Clear-Host
    Show-Banner
    Show-Menu

    $choice = Read-Host "Enter choice"

    switch ($choice.ToUpper()) {
        "R" { Run-Profile }
        "T" { Start-TensorBoard }
        "Q" {
            Write-Host ""
            exit 0
        }
        default {
            Write-Host ""
            Write-Host "Invalid choice" -ForegroundColor Red
            Read-Host "Press Enter to continue"
        }
    }
}
