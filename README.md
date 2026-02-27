# MBIRJAX Profiler

Python profiling of MBIRJAX demo workflows in Docker. Captures cProfile function timing and XLA computation graphs (HLO dumps) for FPGA optimization analysis.

## Prerequisites

- **Docker** (Docker Desktop on Windows/macOS, or Docker Engine on Linux)
- **NVIDIA GPU** with drivers 565.90+ and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

## Quick Start

```bash
# Linux / macOS
./profile.sh demo_1_shepp_logan.py

# Windows (PowerShell)
.\profile.ps1 demo_1_shepp_logan.py
```

This builds the image (if needed), profiles the specified demo, and saves results to `./output/profiles/`.

## Usage

### PowerShell (Windows)

```powershell
.\profile.ps1 demo_1_shepp_logan.py              # VCD Reconstruction (parallel beam)
.\profile.ps1 demo_2_large_object.py             # Large object (partial projection)
.\profile.ps1 demo_3_cropped_center_recon.py     # Cropped center reconstruction
.\profile.ps1 demo_4_wrong_rotation_direction.py # Wrong rotation direction (cone beam)
.\profile.ps1 demo_5_fbp_fdk.py                  # FBP Reconstruction
.\profile.ps1 demo_6_qggmrf_denoiser.py          # QGGMRF Denoising
```

### Bash (Linux / macOS)

```bash
./profile.sh demo_1_shepp_logan.py              # VCD Reconstruction (parallel beam)
./profile.sh demo_2_large_object.py             # Large object (partial projection)
./profile.sh demo_3_cropped_center_recon.py     # Cropped center reconstruction
./profile.sh demo_4_wrong_rotation_direction.py # Wrong rotation direction (cone beam)
./profile.sh demo_5_fbp_fdk.py                  # FBP Reconstruction
./profile.sh demo_6_qggmrf_denoiser.py          # QGGMRF Denoising
```

### Build images manually

```bash
docker compose build cpu        # CPU-only image
docker compose build gpu        # GPU image (requires NVIDIA GPU)
```

### Manual docker commands (CI / advanced)

```bash
docker compose run --rm cpu python /scripts/profiling_wrapper.py /demos/demo_1_shepp_logan.py
docker compose run --rm gpu python /scripts/profiling_wrapper.py /demos/demo_6_qggmrf_denoiser.py
```

## What It Does

Runs each demo script with cProfile to capture Python function-level timing. Also generates XLA computation graphs (HLO dumps) showing the operations that could be mapped to FPGA.

The original demo scripts from the MBIRJAX repository are executed unmodified via `runpy.run_path()`. Only GUI functions (`slice_viewer`, `easygui`) are mocked out for headless operation.

## Demos Profiled

| Demo | Script | Core Operation | Default Parameters |
|------|--------|----------------|-------------------|
| 1 | `demo_1_shepp_logan.py` | VCD Reconstruction (`ct_model.recon`) | parallel beam, 64 views, 40 rows, 128 channels |
| 2 | `demo_2_large_object.py` | VCD Reconstruction (`ct_model.recon`) | parallel beam, 120 views, 80 rows, 100 channels |
| 3 | `demo_3_cropped_center_recon.py` | Cropped Center Recon (`ct_model.recon`) | parallel beam, 400 views, 20 rows, 400 channels |
| 4 | `demo_4_wrong_rotation_direction.py` | VCD Reconstruction (`ct_model.recon`) | cone beam, 64 views, 40 rows, 128 channels |
| 5 | `demo_5_fbp_fdk.py` | FBP Reconstruction (`ct_model.direct_recon`) | parallel beam, 128 views, 128 rows, 128 channels |
| 6 | `demo_6_qggmrf_denoiser.py` | QGGMRF Denoising (`denoiser.denoise`) | 100x100x100, sigma=[0.05, 0.1, 0.15], seed=42 |

## Output

```
output/
  profiles/
    demo_1_shepp_logan_python.txt   # cProfile text summary (top 30 functions)
    demo_1_shepp_logan_python.prof  # cProfile binary (for snakeviz)
  hlo_dumps_xla/                    # XLA computation graphs for FPGA analysis
    *.html                          # Interactive graph visualizations
    jit_*/                          # Per-function compilation data
```

## Viewing Results

### Python profile - Text (`*_python.txt`)

```powershell
Get-Content .\output\profiles\demo_1_shepp_logan_python.txt
```

Shows cumulative time per function. Look for hotspots in `mbirjax` and `jax` calls.

### Python profile - Interactive (`*_python.prof`)

```powershell
pip install snakeviz
snakeviz .\output\profiles\demo_1_shepp_logan_python.prof
```

Opens an interactive flame graph in your browser.

### HLO graphs (`hlo_dumps_xla/`)

```powershell
# Open interactive HTML visualization
explorer .\output\hlo_dumps_xla\
# Then double-click any .html file
```

The HTML files show XLA computation graphs with clickable nodes showing operations (matrix multiplies, reductions, etc.) that could be mapped to FPGA.

## XLA Flags

Set in the Dockerfile as environment variables, applied to all profiling runs.

| Flag | Purpose |
|------|---------|
| `--xla_dump_to` | Dump HLO computation graphs |
| `--xla_dump_hlo_as_text` | Human-readable HLO text format |
| `--xla_dump_hlo_as_html` | Interactive HTML graph visualization |
| `--xla_hlo_profile` | Per-op profiling in HLO graph (GPU only) |
| `--xla_gpu_deterministic_ops` | Deterministic GPU operations |

## Project Structure

```
mbirjax-profiler/
  profile.sh              # Linux/macOS launcher
  profile.ps1             # Windows launcher
  Dockerfile              # GPU (CUDA 12.8) and CPU (Ubuntu 22.04) modes
  docker-compose.yml      # gpu, cpu services
  demos/
    demo_1_shepp_logan.py             # Parallel beam VCD reconstruction
    demo_2_large_object.py            # Large object partial projection
    demo_3_cropped_center_recon.py    # Cropped center reconstruction
    demo_4_wrong_rotation_direction.py # Wrong rotation direction
    demo_5_fbp_fdk.py                 # FBP/FDK reconstruction
    demo_6_qggmrf_denoiser.py         # QGGMRF denoising
  scripts/
    profiling_wrapper.py         # Profiling wrapper (cProfile)
  output/                        # Generated at runtime (not in git)
```

## Known Limitations

- **GUI is mocked** — `easygui` and `mbirjax.slice_viewer` are patched out for headless Docker operation. The demos run to completion without displaying visualizations.
