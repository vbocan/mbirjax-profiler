# MBIRJAX Profiler

CPU profiling tool for MBIRJAX reconstruction algorithms.

## Purpose

This profiler measures the computational cost of MBIRJAX operations across different geometries, algorithms, and problem sizes. It profiles:

- ParallelBeamModel and ConeBeamModel
- MBIR, FBP, and FDK reconstruction algorithms
- Forward and back-projection operations
- Regularization computations
- Scaling behavior across different volume sizes

Output includes timing data, call counts, and performance analysis.

## Quick Start

```bash
./start.ps1
```

Select a profiling preset:
- **[1] Small** - Quick test (5 minutes)
- **[2] Medium** - Standard profile (30 minutes) - **RECOMMENDED**
- **[3] Large** - Complete analysis (2+ hours)

Results are saved to `output/mbirjax_profile_*.json`

## Running via Docker

```bash
# Quick profile
docker-compose run --rm mbirjax-profiler \
  python /scripts/comprehensive_profiler.py --preset small

# Standard profile (recommended)
docker-compose run --rm mbirjax-profiler \
  python /scripts/comprehensive_profiler.py --preset medium

# Complete profile
docker-compose run --rm mbirjax-profiler \
  python /scripts/comprehensive_profiler.py --preset large
```

## Output

After profiling completes:

- **JSON Data**: `output/mbirjax_profile_*.json` - Detailed timing for AI analysis
- **Binary Profile**: `output/mbirjax_profile_*.prof` - For snakeviz visualization
- **Console**: Summary of operation timings

## Analyzing Results

The JSON output is structured for AI analysis and includes:
- Timings grouped by operation type and geometry
- Scaling analysis (complexity estimates)
- Summary with slowest operations and category totals

Feed the JSON file to an AI assistant for FPGA implementation recommendations.

## Project Structure

```
scripts/
  └── comprehensive_profiler.py    Main profiler

output/                            Profiling results
  ├── mbirjax_profile_*.prof       Binary profile (for snakeviz)
  └── mbirjax_profile_*.json       Timing data (for AI analysis)
```

## System Requirements

- Docker & Docker Compose
- 8+ GB RAM (for 512³ volumes)
- Multi-core CPU
- 10 GB disk space for output

## Next Steps

1. Run profiler: `./start.ps1` → select [2]
2. Visualize with snakeviz: `./start.ps1` → select [V]
3. Feed JSON to AI for FPGA implementation analysis
