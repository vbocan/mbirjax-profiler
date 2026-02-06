#Requires -Version 5.1
<#
.SYNOPSIS
    MBIRJAX Profiler - Raw timing data collection for FPGA analysis

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
    Write-Host "  MBIRJAX Profiler" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Menu {
    Write-Host "  [R] Run profiler" -ForegroundColor White
    Write-Host "  [V] View profile with snakeviz" -ForegroundColor White
    Write-Host "  [Q] Quit" -ForegroundColor White
    Write-Host ""
}

function Run-Profile {
    Write-Host ""
    Write-Host "Running profiler..." -ForegroundColor Cyan
    Write-Host "Volume sizes: 32, 64, 128, 256" -ForegroundColor Gray
    Write-Host "Runs per size: 3" -ForegroundColor Gray
    Write-Host ""

    Set-Location $PSScriptRoot
    docker-compose run --rm mbirjax-profiler python /scripts/comprehensive_profiler.py

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[OK] Profiling completed" -ForegroundColor Green
        Write-Host ""
        Write-Host "Output:" -ForegroundColor Cyan
        Write-Host "  JSON: output/mbirjax_profile_*.json" -ForegroundColor Gray
        Write-Host "  Prof: output/mbirjax_profile_*.prof" -ForegroundColor Gray
        Write-Host ""
        Read-Host "Press Enter to continue"
    } else {
        Write-Host ""
        Write-Host "[FAIL] Profiling failed" -ForegroundColor Red
        Read-Host "Press Enter to continue"
    }
}

function View-Profile {
    $profileFiles = Get-ChildItem -Path "output" -Filter "*.prof" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    if ($profileFiles.Count -eq 0) {
        Write-Host ""
        Write-Host "No profile files found. Run profiler first." -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    Write-Host ""
    Write-Host "Available profiles:" -ForegroundColor Cyan
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

    Write-Host ""
    Write-Host "Starting snakeviz..." -ForegroundColor Cyan

    Set-Location $PSScriptRoot
    $outputDir = (Resolve-Path ".\output").Path
    $dockerPath = "/output/$($profileFile.Name)"
    $snakevizUrl = "http://localhost:8080/snakeviz/$($dockerPath)"

    # Kill any existing containers using port 8080
    docker kill $(docker ps -q --filter "publish=8080") 2>$null | Out-Null
    Start-Sleep -Milliseconds 500

    $proc = Start-Process -FilePath "docker" -ArgumentList `
        "run", "--rm", "-p", "8080:8080", `
        "-v", "${outputDir}:/output", `
        "mbirjax-profiler:latest", `
        "snakeviz", "-s", "-H", "0.0.0.0", "-p", "8080", $dockerPath `
        -PassThru

    Start-Sleep -Seconds 3

    Write-Host ""
    Write-Host "snakeviz ready: $snakevizUrl" -ForegroundColor Green
    Write-Host ""

    try { Start-Process $snakevizUrl } catch { }

    Write-Host "Close Docker window when done." -ForegroundColor Gray
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
