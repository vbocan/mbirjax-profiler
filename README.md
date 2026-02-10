# MBIRJAX GPU Profiler

XLA-level GPU profiling of all MBIRJAX operations for FPGA candidate identification. Uses `jax.profiler.trace` to capture GPU execution timelines viewable in TensorBoard/XProf, plus XLA cost analysis and HLO computation graph dumps.

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
# Profile
docker compose run --rm mbirjax-profiler python /scripts/comprehensive_profiler.py

# View traces in TensorBoard
docker compose run --rm -p 6006:6006 mbirjax-profiler tensorboard --logdir=/output/jax_traces --host=0.0.0.0 --port=6006
```

Then open http://localhost:6006.

## What It Does

Profiles every MBIRJAX operation across multiple volume sizes (32^3, 64^3, 128^3, 256^3) with 3 runs per size:

- **Run 1**: JIT warmup (XLA compilation happens here)
- **Run 2**: Traced via `jax.profiler.trace` (captures XLA execution timeline)
- **Run 3**: Timing only (for comparison)

After timing runs, collects XLA cost analysis (FLOPs, bytes accessed) and dumps HLO computation graphs for key operations.

**Operations profiled (28):**
- ParallelBeamModel: forward_project, back_project, sparse variants, hessian_diagonal, direct_filter, mbir_recon, prox_map, fbp_recon, fbp_filter, direct_recon, weight generation
- ConeBeamModel: forward_project, back_project, sparse variants, hessian_diagonal, direct_filter, mbir_recon, prox_map, fdk_recon, fdk_filter, direct_recon, weight generation
- QGGMRFDenoiser: denoise
- Utilities: median_filter3d, gen_pixel_partition variants

## Output

Each profiling run produces:

| Output | Location | Purpose |
|--------|----------|---------|
| Timing + cost analysis JSON | `output/mbirjax_profile_*.json` | Wall-clock timing, XLA FLOPs/bytes estimates |
| XLA traces | `output/jax_traces/<timestamp>/vol<N>/` | TensorBoard/XProf GPU execution timeline |
| HLO dumps | `output/hlo_dumps/<timestamp>/` | XLA computation graphs per operation |

### XProf tools to use

| Tool | What It Shows | FPGA Relevance |
|------|---------------|----------------|
| Trace Viewer | GPU compute vs memory transfer timeline | Identifies memory transfer bottlenecks |
| Roofline Analysis | Arithmetic intensity per operation | Classifies compute-bound vs memory-bound ops |
| HLO Op Stats | Per-operation time, GFLOPS/s, bandwidth | Ranks operations by cost |
| GPU Kernel Stats | Per-kernel metrics mapped to JAX ops | Ground truth kernel timing |
| Memory Viewer | Buffer lifetimes and peak allocation | FPGA on-chip memory sizing |

### JSON structure

```json
{
  "environment": {
    "backend": "gpu",
    "devices": ["cuda:0"],
    "jax_version": "...",
    "mbirjax_version": "..."
  },
  "measurements": [
    {"operation": "parallel_forward_project", "volume_size": 64, "run": 1, "time": 0.234, "traced": false}
  ],
  "cost_analysis": {
    "64": {
      "parallel_forward_project": {"flops": 123456, "bytes accessed": 78900}
    }
  }
}
```

## XLA HLO Graph Visualization

The profiler produces HLO graphs at two levels:

1. **Per-operation HLO text** (`output/hlo_dumps/<timestamp>/`) — programmatic dumps from the profiler script, one file per operation per volume size. Grep-able and diff-able.

2. **Comprehensive XLA HTML graphs** (`output/hlo_dumps_xla/`) — produced automatically via `XLA_FLAGS`. These are interactive HTML files showing the full computation graph with zoom/pan. Open any `.html` file in a browser to explore the graph visually.

The comprehensive dumps include every XLA compilation pass (before and after optimization), so the output is large. To disable, comment out the `XLA_FLAGS` line in `docker-compose.yml`.

## Verification

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
