#Requires -Version 5.1
<#
.SYNOPSIS
    MBIRJAX Profiler - Interactive cProfile Analysis for FPGA Optimization

.DESCRIPTION
    Interactive menu-driven profiler for MBIRJAX reconstruction analysis.
    Profiles operations and visualizes results with snakeviz.

.EXAMPLE
    .\start.ps1
#>

$ErrorActionPreference = "Stop"

# Configuration
$script:ImageSource = "local"  # "local" or "ghcr"
$script:GhcrImage = "ghcr.io/mbirjax-profiler:latest"

# Validate Docker
try {
    $null = docker info 2>&1 | Out-Null
} catch {
    Write-Host ""
    Write-Host "✗ Error: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

function Show-Banner {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║  MBIRJAX Profiler - cProfile Analysis for FPGA Optimization║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Menu {
    # Show current source
    $sourceDisplay = if ($script:ImageSource -eq "ghcr") {
        "GitHub Container Registry"
    } else {
        "Local Docker Build"
    }
    Write-Host "Current source: $sourceDisplay" -ForegroundColor Gray
    Write-Host ""

    Write-Host "Select an option:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Image Source:" -ForegroundColor Gray
    Write-Host "    [S] Switch source (currently: $($script:ImageSource.ToUpper()))" -ForegroundColor White
    Write-Host ""
    Write-Host "  Comprehensive Feature Profiling:" -ForegroundColor Gray
    Write-Host "    [1] Small    - 64³, 128³ volumes (5 min, all geometries)" -ForegroundColor White
    Write-Host "    [2] Medium   - 64³, 128³, 256³ volumes (30 min, all geometries) - RECOMMENDED" -ForegroundColor Cyan
    Write-Host "    [3] Large    - 256³+ volumes (2+ hours, complete analysis)" -ForegroundColor White
    Write-Host ""
    Write-Host "  Analysis:" -ForegroundColor Gray
    Write-Host "    [A] Analyze profile results (hotspots, function stats)" -ForegroundColor White
    Write-Host ""
    Write-Host "  Visualization:" -ForegroundColor Gray
    Write-Host "    [V] View profile with snakeviz (select from list)" -ForegroundColor White
    Write-Host ""
    Write-Host "  Exit:" -ForegroundColor Gray
    Write-Host "    [Q] Quit" -ForegroundColor White
    Write-Host ""
}

function Run-Comprehensive-Profile {
    param(
        [string]$Preset = "medium"
    )

    Write-Host ""
    Write-Host "Running Comprehensive Feature Profiler..." -ForegroundColor Cyan
    Write-Host "Preset: $Preset" -ForegroundColor Gray

    $sourceDisplay = if ($script:ImageSource -eq "ghcr") {
        "GitHub Container Registry"
    } else {
        "Local Docker Build"
    }
    Write-Host "Source: $sourceDisplay" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Features profiled:" -ForegroundColor Cyan
    Write-Host "  • ParallelBeamModel (2D projections)" -ForegroundColor Gray
    Write-Host "  • ConeBeamModel (3D cone geometry)" -ForegroundColor Gray
    Write-Host "  • MBIR reconstruction" -ForegroundColor Gray
    Write-Host "  • FBP/FDK direct reconstruction" -ForegroundColor Gray
    Write-Host "  • Scaling analysis (volume sizes)" -ForegroundColor Gray
    Write-Host ""

    Set-Location $PSScriptRoot
    $outputDir = (Resolve-Path ".\output").Path

    if ($script:ImageSource -eq "ghcr") {
        # Pull from GitHub Container Registry
        Write-Host "Pulling image from GitHub Container Registry..." -ForegroundColor Gray
        docker run --rm `
            -v "${outputDir}:/output" `
            $script:GhcrImage `
            python /scripts/comprehensive_profiler.py --preset $Preset
    } else {
        # Use local docker-compose build
        docker-compose run --rm mbirjax-profiler python /scripts/comprehensive_profiler.py `
            --preset $Preset
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Comprehensive profiling completed!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Output files:" -ForegroundColor Cyan
        Write-Host "  • JSON data: output/comprehensive_profile_*.json" -ForegroundColor Gray
        Write-Host "  • Binary profiles: output/logs/profiles/" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "  1. Use option [V] to view detailed profiles with snakeviz" -ForegroundColor White
        Write-Host "  2. Review JSON output for all feature timings" -ForegroundColor White
        Write-Host ""
        Read-Host "Press Enter to continue"
    } else {
        Write-Host ""
        Write-Host "✗ Profiling failed" -ForegroundColor Red
        Read-Host "Press Enter to continue"
    }
}


function View-Profile {
    # Find all .prof files (binary cProfile output)
    $profileFiles = Get-ChildItem -Path "output" -Filter "*.prof" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "_profile\.txt" } |
        Sort-Object LastWriteTime -Descending

    if ($profileFiles.Count -eq 0) {
        Write-Host ""
        Write-Host "✗ No profile files found in output/ directory" -ForegroundColor Red
        Write-Host ""
        Write-Host "Run a profile first with options [1], [2], [3], or [C]" -ForegroundColor Gray
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    # Display list of profiles for selection
    Write-Host ""
    Write-Host "Available profiles:" -ForegroundColor Cyan
    Write-Host ""

    for ($i = 0; $i -lt $profileFiles.Count; $i++) {
        $prof = $profileFiles[$i]
        $size = [math]::Round($prof.Length / 1KB, 1)
        $date = $prof.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "  [$($i + 1)] $($prof.Name)" -ForegroundColor White
        Write-Host "      $size KB  |  $date" -ForegroundColor Gray
    }

    Write-Host ""
    $selection = Read-Host "Select profile number"

    # Validate selection
    if (-not [int]::TryParse($selection, [ref]$null) -or [int]$selection -lt 1 -or [int]$selection -gt $profileFiles.Count) {
        Write-Host ""
        Write-Host "✗ Invalid selection" -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    $profileFile = $profileFiles[[int]$selection - 1]

    Write-Host ""
    Write-Host "Starting snakeviz viewer..." -ForegroundColor Cyan
    Write-Host "Profile: $($profileFile.Name)" -ForegroundColor Gray
    Write-Host ""

    Set-Location $PSScriptRoot

    # Get absolute path for volume mount (Docker on Windows needs this format)
    $outputDir = (Resolve-Path ".\output").Path

    # Construct the proper snakeviz URL
    $profilePath = "/output/$($profileFile.Name)"
    $snakevizUrl = "http://localhost:8080/snakeviz/$($profilePath)"

    # Kill any existing containers using port 8080
    Write-Host "Cleaning up previous sessions..." -ForegroundColor Gray
    docker kill $(docker ps -q --filter "publish=8080") 2>$null | Out-Null
    Start-Sleep -Milliseconds 500

    Write-Host "Starting snakeviz container..." -ForegroundColor Gray

    # Start snakeviz in Docker
    # Note: -H 0.0.0.0 makes snakeviz listen on all interfaces (needed for Docker port mapping)
    # Using a separate window for the container so user can see logs
    $imageToUse = if ($script:ImageSource -eq "ghcr") { $script:GhcrImage } else { "mbirjax-profiler:latest" }
    $proc = Start-Process -FilePath "docker" -ArgumentList `
        "run", "--rm", "-p", "8080:8080", `
        "-v", "${outputDir}:/output", `
        $imageToUse, `
        "snakeviz", "-s", "-H", "0.0.0.0", "-p", "8080", "/output/$($profileFile.Name)" `
        -PassThru

    Write-Host "Container started. Waiting for snakeviz to bind to port 8080..." -ForegroundColor Gray
    Start-Sleep -Seconds 3

    Write-Host ""
    Write-Host "snakeviz is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "URL: $snakevizUrl" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Opening in your default browser..." -ForegroundColor Yellow
    Write-Host ""

    # Open browser to the snakeviz URL
    try {
        Start-Process $snakevizUrl
    } catch {
        Write-Host "Could not auto-open browser. Please visit:" -ForegroundColor Yellow
        Write-Host "  $snakevizUrl" -ForegroundColor Cyan
    }

    Write-Host "snakeviz is running. Close the Docker window when you're done." -ForegroundColor Gray
    Write-Host ""
    Read-Host "Press Enter to continue"

    Write-Host ""
    Write-Host "Viewer stopped" -ForegroundColor Cyan
    Write-Host ""

    # Cleanup (snakeviz logs are stored in output/logs/snakeviz)
    if (Test-Path "output/logs/snakeviz/snakeviz-output.log") {
        Remove-Item "output/logs/snakeviz/snakeviz-output.log" -ErrorAction SilentlyContinue
    }

    Read-Host "Press Enter to continue"
}

function Analyze-Profile {
    # Find all profile text files (cProfile output)
    $profileFiles = Get-ChildItem -Path "output" -Filter "*_profile.txt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending

    if ($profileFiles.Count -eq 0) {
        Write-Host ""
        Write-Host "✗ No profile results found in output/" -ForegroundColor Red
        Write-Host ""
        Write-Host "Run a profile first with options [1], [2], or [3]" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    # Display list of profiles for selection
    Write-Host ""
    Write-Host "Available profiles:" -ForegroundColor Cyan
    Write-Host ""

    for ($i = 0; $i -lt $profileFiles.Count; $i++) {
        $prof = $profileFiles[$i]
        $size = [math]::Round($prof.Length / 1KB, 1)
        $date = $prof.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        Write-Host "  [$($i + 1)] $($prof.Name)" -ForegroundColor White
        Write-Host "      $size KB  |  $date" -ForegroundColor Gray
    }

    Write-Host ""
    $selection = Read-Host "Select profile number"

    # Validate selection
    if (-not [int]::TryParse($selection, [ref]$null) -or [int]$selection -lt 1 -or [int]$selection -gt $profileFiles.Count) {
        Write-Host ""
        Write-Host "✗ Invalid selection" -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue"
        return
    }

    $profileFile = $profileFiles[[int]$selection - 1]

    Write-Host ""
    Write-Host "Analyzing: $($profileFile.Name)" -ForegroundColor Cyan
    Write-Host ""

    Set-Location $PSScriptRoot
    $outputDir = (Resolve-Path ".\output").Path

    if ($script:ImageSource -eq "ghcr") {
        # Pull from GitHub Container Registry
        docker run --rm `
            -v "${outputDir}:/output" `
            $script:GhcrImage `
            python /scripts/analyze_profile.py "/output/$($profileFile.Name)"
    } else {
        # Use local docker-compose build
        docker-compose run --rm mbirjax-profiler python /scripts/analyze_profile.py "/output/$($profileFile.Name)"
    }

    Write-Host ""
    Read-Host "Press Enter to continue"
}

function List-Profiles {
    Write-Host ""
    Write-Host "Recent Profiles:" -ForegroundColor Cyan
    Write-Host "-" * 80

    $profiles = Get-ChildItem -Path "output" -Filter "*.prof" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notmatch "_profile\.txt" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 10

    if ($profiles.Count -eq 0) {
        Write-Host "No profiles found" -ForegroundColor Gray
    } else {
        Write-Host ""
        foreach ($prof in $profiles) {
            $size = [math]::Round($prof.Length / 1KB, 0)
            $date = $prof.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
            $name = $prof.Name -replace "\.prof", ""
            Write-Host "  $name"
            Write-Host "    $size KB  |  $date" -ForegroundColor Gray
        }
        Write-Host ""
    }

    Write-Host ""
    Read-Host "Press Enter to continue"
}

function Switch-ImageSource {
    Write-Host ""
    Write-Host "Select image source:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  [1] Local Docker Build (docker-compose)" -ForegroundColor White
    Write-Host "  [2] GitHub Container Registry (GHCR)" -ForegroundColor White
    Write-Host ""

    $choice = Read-Host "Enter your choice"

    switch ($choice) {
        "1" {
            $script:ImageSource = "local"
            Write-Host ""
            Write-Host "✓ Switched to Local Docker Build" -ForegroundColor Green
            Write-Host ""
            Write-Host "The profiler will build and run from your local Docker setup." -ForegroundColor Gray
            Write-Host ""
        }
        "2" {
            $script:ImageSource = "ghcr"
            Write-Host ""
            Write-Host "✓ Switched to GitHub Container Registry" -ForegroundColor Green
            Write-Host ""
            Write-Host "The profiler will pull and run from: $($script:GhcrImage)" -ForegroundColor Gray
            Write-Host ""
            Write-Host "Note: Requires GitHub Container Registry to have pushed the image." -ForegroundColor Yellow
            Write-Host ""
        }
        default {
            Write-Host ""
            Write-Host "✗ Invalid choice" -ForegroundColor Red
            Write-Host ""
        }
    }

    Read-Host "Press Enter to continue"
}

# Main loop
Set-Location $PSScriptRoot

while ($true) {
    Clear-Host
    Show-Banner
    Show-Menu

    $choice = Read-Host "Enter your choice"

    switch ($choice.ToUpper()) {
        "S" {
            Switch-ImageSource
        }
        "1" {
            Run-Comprehensive-Profile -Preset "small"
        }
        "2" {
            Run-Comprehensive-Profile -Preset "medium"
        }
        "3" {
            Run-Comprehensive-Profile -Preset "large"
        }
        "A" {
            Analyze-Profile
        }
        "V" {
            View-Profile
        }
        "Q" {
            Write-Host ""
            Write-Host "Goodbye!" -ForegroundColor Cyan
            exit 0
        }
        default {
            Write-Host ""
            Write-Host "✗ Invalid choice" -ForegroundColor Red
            Read-Host "Press Enter to continue"
        }
    }
}
