# MBIRJAX Profiler
#
# Usage:
#   .\profile.ps1 demo_1_shepp_logan.py
#
# Output: ./output/profiles/

param(
    [Parameter(Position=0)]
    [string]$Demo
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Show help if no parameters
if (-not $Demo) {
    Write-Host "MBIRJAX Profiler" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\profile.ps1 <demo_script>" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available demos:"
    Write-Host "  demo_1_shepp_logan.py              VCD Reconstruction (parallel beam)"
    Write-Host "  demo_2_large_object.py             Large object (partial projection)"
    Write-Host "  demo_3_cropped_center_recon.py     Cropped center reconstruction"
    Write-Host "  demo_4_wrong_rotation_direction.py Wrong rotation direction (cone beam)"
    Write-Host "  demo_5_fbp_fdk.py                  FBP Reconstruction"
    Write-Host "  demo_6_qggmrf_denoiser.py          QGGMRF Denoising"
    Write-Host ""
    Write-Host "Example:"
    Write-Host "  .\profile.ps1 demo_1_shepp_logan.py" -ForegroundColor Green
    Write-Host ""
    Write-Host "Output:"
    Write-Host "  ./output/profiles/*_python.txt     cProfile text summary"
    Write-Host "  ./output/profiles/*_python.prof    cProfile binary (snakeviz)"
    Write-Host "  ./output/hlo_dumps_xla/*.html      XLA computation graphs"
    exit 0
}

$DemoPath = "/demos/$Demo"

# Build image if needed
$Image = "mbirjax-profiler:gpu"
if (-not (docker images -q $Image 2>$null)) {
    Write-Host "Building $Image..." -ForegroundColor Yellow
    docker compose build gpu
}

# Run profiler
Write-Host "Profiling: $DemoPath" -ForegroundColor Cyan
docker compose run --rm gpu python /scripts/profiling_wrapper.py $DemoPath

Write-Host "`nOutput: ./output/profiles/" -ForegroundColor Green
