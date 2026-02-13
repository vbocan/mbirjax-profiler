# MBIRJAX Profiler — Windows launcher
# Provides menu-driven access to GPU/CPU profiling and TensorBoard visualization
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$IMAGE_GPU   = "mbirjax-profiler:gpu"
$IMAGE_CPU   = "mbirjax-profiler:cpu"
$OUTPUT_DIR  = Join-Path $PSScriptRoot "output"

function Write-Header {
    Clear-Host
    Write-Host ""
    Write-Host "  MBIRJAX Profiler" -ForegroundColor Cyan
    Write-Host "  ================" -ForegroundColor DarkCyan
    Write-Host ""
}

function Test-DockerImage {
    param([string]$Image)
    $result = docker images -q $Image 2>$null
    return [bool]$result
}

function Build-Image {
    param([string]$Mode)

    Write-Host ""
    if ($Mode -eq "gpu") {
        Write-Host "  Building GPU image..." -ForegroundColor Yellow
        docker build -t $IMAGE_GPU .
    } else {
        Write-Host "  Building CPU image..." -ForegroundColor Yellow
        docker build `
            --build-arg BASE_IMAGE=ubuntu:22.04 `
            --build-arg JAX_PACKAGE=jax `
            --build-arg JAX_PLATFORMS_DEFAULT=cpu `
            --build-arg XLA_FLAGS_DEFAULT="--xla_dump_to=/output/hlo_dumps_xla --xla_dump_hlo_as_text --xla_dump_hlo_as_html=true" `
            -t $IMAGE_CPU .
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Build failed." -ForegroundColor Red
        return $false
    }
    Write-Host "  Build complete." -ForegroundColor Green
    return $true
}

function Ensure-Image {
    param([string]$Mode)

    $image = if ($Mode -eq "gpu") { $IMAGE_GPU } else { $IMAGE_CPU }

    if (Test-DockerImage $image) {
        return $true
    }

    Write-Host "  Image '$image' not found." -ForegroundColor Yellow
    $reply = Read-Host "  Build it now? [Y/n]"
    if ($reply -match "^[nN]") {
        return $false
    }
    return (Build-Image $Mode)
}

function Run-Profiler {
    param([string]$Mode)

    if (-not (Ensure-Image $Mode)) { return }

    if (-not (Test-Path $OUTPUT_DIR)) {
        New-Item -ItemType Directory -Path $OUTPUT_DIR | Out-Null
    }

    Write-Host ""
    Write-Host "  Running profiler ($Mode mode)..." -ForegroundColor Cyan
    Write-Host ""

    if ($Mode -eq "gpu") {
        docker run --rm --gpus all --shm-size=16g `
            --cap-add=SYS_ADMIN `
            --security-opt=seccomp:unconfined `
            --ulimit memlock=-1:-1 `
            -v "${OUTPUT_DIR}:/output" `
            $IMAGE_GPU
    } else {
        docker run --rm `
            -v "${OUTPUT_DIR}:/output" `
            $IMAGE_CPU
    }

    Write-Host ""
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Profiling complete. Results in: $OUTPUT_DIR" -ForegroundColor Green
    } else {
        Write-Host "  Profiler exited with errors." -ForegroundColor Red
    }
}

function Start-TensorBoard {
    # Use whichever image is available (prefer GPU)
    $image = $null
    if (Test-DockerImage $IMAGE_GPU) {
        $image = $IMAGE_GPU
    } elseif (Test-DockerImage $IMAGE_CPU) {
        $image = $IMAGE_CPU
    }

    if (-not $image) {
        Write-Host "  No profiler image found. Build one first (option B)." -ForegroundColor Red
        return
    }

    $tracesDir = Join-Path $OUTPUT_DIR "jax_traces"
    if (-not (Test-Path $tracesDir)) {
        Write-Host "  No traces found in $tracesDir" -ForegroundColor Red
        Write-Host "  Run the profiler first to generate traces." -ForegroundColor Yellow
        return
    }

    Write-Host ""
    Write-Host "  Starting TensorBoard on http://localhost:6006" -ForegroundColor Cyan
    Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
    Write-Host ""

    docker run --rm -p 6006:6006 `
        -v "${OUTPUT_DIR}:/output" `
        $image `
        tensorboard --logdir=/output/jax_traces --host=0.0.0.0
}

# --- Main loop ---
while ($true) {
    Write-Header

    $gpuStatus = if (Test-DockerImage $IMAGE_GPU) { "ready" } else { "not built" }
    $cpuStatus = if (Test-DockerImage $IMAGE_CPU) { "ready" } else { "not built" }

    Write-Host "  Images:  GPU [$gpuStatus]  CPU [$cpuStatus]" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  [B] Build images" -ForegroundColor Yellow
    Write-Host "  [G] Profile  (GPU)" -ForegroundColor Green
    Write-Host "  [C] Profile  (CPU)" -ForegroundColor Green
    Write-Host "  [V] View results  (TensorBoard)" -ForegroundColor Magenta
    Write-Host "  [Q] Quit" -ForegroundColor DarkGray
    Write-Host ""

    $choice = Read-Host "  Choice"

    switch ($choice.ToUpper()) {
        "G" { Run-Profiler "gpu" }
        "C" { Run-Profiler "cpu" }
        "V" { Start-TensorBoard }
        "B" {
            Write-Host ""
            Write-Host "  [1] GPU  [2] CPU  [3] Both" -ForegroundColor Yellow
            $sub = Read-Host "  Which"
            switch ($sub) {
                "1" { Build-Image "gpu" }
                "2" { Build-Image "cpu" }
                "3" { Build-Image "gpu"; Build-Image "cpu" }
                default { Write-Host "  Invalid choice." -ForegroundColor Red }
            }
        }
        "Q" { Write-Host ""; break }
        default { Write-Host "  Invalid choice." -ForegroundColor Red }
    }

    if ($choice.ToUpper() -eq "Q") { break }

    Write-Host ""
    Read-Host "  Press Enter to continue"
}
