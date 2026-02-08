# MBIRJAX Profiler

GPU-accelerated profiling of all MBIRJAX operations using Scalene for line-level CPU/GPU/memory analysis, targeting FPGA candidate identification.

## Prerequisites

- **Docker Desktop** with Docker Compose V2 (`docker compose`, not `docker-compose`)
- **NVIDIA GPU** with driver 565.90+ (RTX 5080 or compatible)
- **NVIDIA Container Toolkit** (included with Docker Desktop on Windows/WSL2)

Verify GPU access in Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

## Quick Start

**Windows:**
```powershell
.\start.ps1
```

**Linux/macOS:**
```bash
chmod +x start.sh
./start.sh
```

Or run directly:
```bash
docker compose run --rm mbirjax-profiler python -m scalene run --gpu -o /output/scalene_profile.json /scripts/comprehensive_profiler.py
```

## What It Does

Profiles every MBIRJAX operation across multiple volume sizes (32^3, 64^3, 128^3, 256^3) with 3 runs each. Scalene wraps the profiling script externally, providing line-level CPU/GPU/memory breakdown with no code changes needed.

**Operations profiled:**
- ParallelBeamModel: forward_project, back_project, hessian_diagonal, mbir_recon, fbp_recon, fbp_filter, direct_recon
- ConeBeamModel: forward_project, back_project, hessian_diagonal, mbir_recon, fdk_recon, fdk_filter
- Utilities: median_filter3d, gen_pixel_partition

## Output

Each profiling run produces three files in `output/`:

| File | Source | Purpose |
|------|--------|---------|
| `scalene_profile_*.html` | Scalene | Interactive CPU/GPU/memory line-level profile (open in browser) |
| `scalene_profile_*.json` | Scalene | Machine-readable Scalene data |
| `mbirjax_profile_*.json` | Script | Per-operation timing with `block_until_ready()` synchronization |

Timing JSON structure:
```json
{
  "environment": {
    "backend": "gpu",
    "devices": ["cuda:0"],
    "jax_version": "...",
    "mbirjax_version": "..."
  },
  "measurements": [
    {"operation": "parallel_forward_project", "volume_size": 64, "run": 1, "time": 0.234}
  ]
}
```

## Why Scalene

- **GPU profiling**: Tracks time spent on GPU vs CPU per line of code
- **Line-level granularity**: Identifies exact lines consuming GPU/CPU/memory resources
- **Memory tracking**: Shows allocation patterns useful for FPGA memory planning
- **Self-contained HTML**: No server needed (unlike snakeviz) — just open in a browser
- **JAX compatible**: v2.1.3+ fixes the JAX hanging issue (plasma-umass/scalene#106)

## Verification

After building, verify the setup:
```bash
# GPU visible in container
docker compose run --rm mbirjax-profiler nvidia-smi

# JAX sees CUDA
docker compose run --rm mbirjax-profiler python -c "import jax; print(jax.devices())"
```

## System Requirements

- NVIDIA GPU (RTX 5080 or compatible)
- CUDA 12.8+ / Driver 565.90+
- Docker Desktop with Compose V2
- 16+ GB RAM (for 256^3 volumes on GPU)
