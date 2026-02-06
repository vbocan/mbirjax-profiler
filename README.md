# MBIRJAX Profiler

Profiles all MBIRJAX operations to collect raw timing data for FPGA candidate analysis.

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
docker-compose run --rm mbirjax-profiler python /scripts/comprehensive_profiler.py
```

## What It Does

Profiles every MBIRJAX operation across multiple volume sizes (32³, 64³, 128³, 256³) with 3 runs each. No configuration needed.

**Operations profiled:**
- ParallelBeamModel: forward_project, back_project, hessian_diagonal, mbir_recon, fbp_recon, fbp_filter, direct_recon
- ConeBeamModel: forward_project, back_project, hessian_diagonal, mbir_recon, fdk_recon, fdk_filter
- Utilities: median_filter3d, gen_pixel_partition

## Output

- **JSON**: `output/mbirjax_profile_*.json` - Raw timing measurements
- **Prof**: `output/mbirjax_profile_*.prof` - For snakeviz visualization

JSON structure:
```json
{
  "measurements": [
    {"operation": "parallel_forward_project", "volume_size": 64, "run": 1, "time": 0.234},
    ...
  ]
}
```

## Visualization

View call tree with snakeviz:

```bash
./start.ps1  # select [V]
```

## Why cProfile Instead of Scalene?

Scalene has a better UI, but it doesn't work reliably with JAX:

- **JAX compatibility**: Scalene [hangs indefinitely](https://github.com/plasma-umass/scalene/issues/106) when profiling JAX code
- **Async execution**: JAX uses lazy execution; we need `block_until_ready()` calls with manual timing to get accurate per-operation measurements
- **Docker issues**: Scalene has [output problems](https://github.com/plasma-umass/scalene/discussions/612) in containers

cProfile + snakeviz gives us reliable operation-level timing for FPGA acceleration analysis.

## System Requirements

- Docker & Docker Compose
- 8+ GB RAM (for 256³ volumes)
- Multi-core CPU
