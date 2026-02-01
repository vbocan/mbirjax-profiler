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

Results are saved to `output/comprehensive_profile_YYYYMMDD_HHMMSS.json`

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

- **JSON Data**: `output/comprehensive_profile_YYYYMMDD_HHMMSS.json` - Detailed timing for all operations
- **Logs**: `output/logs/` - Profile logs and binary profile data
- **Console**: Summary of operation timings

## Analyzing Results

```bash
docker-compose run --rm mbirjax-profiler \
  python /scripts/analyze_profile.py /output/comprehensive_profile_*.json
```

Shows hotspots, function statistics, and candidate operations for optimization.

## Project Structure

```
scripts/
  ├── comprehensive_profiler.py    Main profiler
  └── analyze_profile.py           Result analysis tool

output/                            Profiling results
  ├── logs/                        Profile logs
  └── comprehensive_profile_*.json  Timing data
```

## System Requirements

- Docker & Docker Compose
- 8+ GB RAM (for 512³ volumes)
- Multi-core CPU
- 10 GB disk space for output

## Next Steps

1. Run profiler: `./start.ps1` → select [2]
2. Review JSON output in `output/`
3. Analyze with: `analyze_profile.py /output/comprehensive_profile_*.json`
