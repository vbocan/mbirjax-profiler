#Requires -Version 5.1
<#
.SYNOPSIS
    MBIRJAX Profiler (Scalene GPU) - Line-level CPU/GPU/memory profiling

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
    Write-Host "  MBIRJAX Profiler (Scalene GPU)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Menu {
    Write-Host "  [R] Run profiler" -ForegroundColor White
    Write-Host "  [V] View Scalene profile" -ForegroundColor White
    Write-Host "  [Q] Quit" -ForegroundColor White
    Write-Host ""
}

function Run-Profile {
    Write-Host ""
    Write-Host "Running Scalene profiler with GPU..." -ForegroundColor Cyan
    Write-Host "Volume sizes: 32, 64, 128, 256" -ForegroundColor Gray
    Write-Host "Runs per size: 3" -ForegroundColor Gray
    Write-Host ""

    Set-Location $PSScriptRoot
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $jsonFile = "scalene_profile_$timestamp.json"
    $htmlFile = "scalene_profile_$timestamp.html"

    # Run Scalene profiler (produces JSON)
    docker compose run --rm mbirjax-profiler python -m scalene run --cpu-only --profile-all --profile-only mbirjax -o "/output/$jsonFile" /scripts/comprehensive_profiler.py

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[FAIL] Profiling failed" -ForegroundColor Red
        Read-Host "Press Enter to continue"
        return
    }

    # Convert JSON profile to self-contained HTML
    Write-Host ""
    Write-Host "Generating HTML report..." -ForegroundColor Cyan
    docker compose run --rm mbirjax-profiler python -m scalene view --standalone "/output/$jsonFile"

    # Rename the generated HTML to include timestamp
    if (Test-Path "output/scalene-profile.html") {
        Move-Item -Force "output/scalene-profile.html" "output/$htmlFile"
    }

    Write-Host ""
    Write-Host "[OK] Profiling completed" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output:" -ForegroundColor Cyan
    Write-Host "  Scalene HTML: output/$htmlFile" -ForegroundColor Gray
    Write-Host "  Scalene JSON: output/$jsonFile" -ForegroundColor Gray
    Write-Host "  Timing JSON:  output/mbirjax_profile_*.json" -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to continue"
}

function View-Profile {
    $profileFiles = Get-ChildItem -Path "output" -Filter "scalene_profile_*.html" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    if ($profileFiles.Count -eq 0) {
        Write-Host ""
        Write-Host "No Scalene HTML profiles found. Run profiler first." -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    Write-Host ""
    Write-Host "Available Scalene profiles:" -ForegroundColor Cyan
    Write-Host ""

    for ($i = 0; $i -lt $profileFiles.Count; $i++) {
        $prof = $profileFiles[$i]
        $size = [math]::Round($prof.Length / 1KB, 0)
        $date = $prof.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
        Write-Host "  [$($i + 1)] $($prof.Name)" -ForegroundColor White
        Write-Host "      $size KB  |  $date" -ForegroundColor Gray
    }

    Write-Host ""
    $selection = Read-Host "Select profile number (or 'q' to cancel)"

    if ($selection -eq 'q') { return }

    if (-not [int]::TryParse($selection, [ref]$null) -or [int]$selection -lt 1 -or [int]$selection -gt $profileFiles.Count) {
        Write-Host ""
        Write-Host "Invalid selection" -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    $profileFile = $profileFiles[[int]$selection - 1]
    $fullPath = (Resolve-Path $profileFile.FullName).Path

    Write-Host ""
    Write-Host "Opening in browser: $($profileFile.Name)" -ForegroundColor Cyan

    try { Start-Process $fullPath } catch {
        Write-Host "Could not open browser. File: $fullPath" -ForegroundColor Yellow
    }

    Write-Host ""
    Read-Host "Press Enter to continue"
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
        "V" { View-Profile }
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
