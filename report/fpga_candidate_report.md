# FPGA Candidate Analysis Report

**Generated:** 2026-02-06
**Profile:** mbirjax_profile_20260206_084448.json
**Environment:** JAX 0.9.0, CPU backend (single-threaded)

## Data Quality Note

Some timing values are negative (likely clock sync issues in Docker). Analysis uses median of valid runs, excluding run 1 which includes JIT compilation overhead.

## Results at 256³ Volume (Largest Size)

| Rank | Operation | Median Time (s) | % of Total | FPGA Priority |
|------|-----------|-----------------|------------|---------------|
| 1 | **cone_mbir_recon** | 85.0 | 48% | **HIGH** |
| 2 | **median_filter3d** | 25.0 | 14% | **HIGH** |
| 3 | **cone_hessian_diagonal** | 19.6 | 11% | **HIGH** |
| 4 | **cone_back_project** | 17.1 | 10% | **HIGH** |
| 5 | **cone_fdk_recon** | 16.2 | 9% | **HIGH** |
| 6 | **cone_forward_project** | 15.9 | 9% | **HIGH** |
| 7 | parallel_mbir_recon | 6.5 | 4% | MEDIUM |
| 8 | parallel_forward_project | 2.1 | 1% | LOW |
| 9 | parallel_back_project | 1.0 | <1% | LOW |
| 10 | parallel_fbp_recon | 0.8 | <1% | LOW |

## Scaling Behavior (64³ → 256³)

| Operation | 64³ Time | 256³ Time | Scale Factor | Pattern |
|-----------|----------|-----------|--------------|---------|
| cone_mbir_recon | 0.16s | 85s | **530×** | Super-linear |
| cone_forward_project | 0.6s | 16s | 27× | ~O(n³) to O(n⁴) |
| cone_back_project | 0.6s | 17s | 28× | ~O(n³) to O(n⁴) |
| cone_hessian_diagonal | 0.26s | 19.6s | 75× | Super-linear |
| median_filter3d | 0.6s | 25s | 42× | ~O(n³) |

## Key Findings

### 1. Cone Beam Dominates

All top 6 operations are cone beam. Parallel beam is approximately 10× faster at the same volume size.

### 2. MBIR Reconstruction is the Bottleneck

`cone_mbir_recon` alone consumes 48% of total profiling time. This is an iterative algorithm that internally calls forward/back projection operations.

### 3. Best FPGA Candidates (Ranked)

| Priority | Function | Reason |
|----------|----------|--------|
| 1 | `forward_project` (cone) | Core primitive, used by MBIR, scales poorly |
| 2 | `back_project` (cone) | Core primitive, used by MBIR, scales poorly |
| 3 | `median_filter3d` | Standalone, embarrassingly parallel, simple algorithm |
| 4 | `hessian_diagonal` (cone) | Used by MBIR, scales poorly |

### 4. Why Not Target MBIR Directly?

MBIR is *iterative* - it's a loop calling forward/back projection repeatedly. Accelerating the primitives (`forward_project`, `back_project`) automatically accelerates MBIR.

## Recommendations

### Immediate Targets

1. **Cone beam forward_project** - Most impactful single function. Ray-tracing through 3D volume with perspective geometry.

2. **Cone beam back_project** - Paired with forward_project, required for any iterative reconstruction.

3. **median_filter3d** - Low-hanging fruit. Simple 3D sliding window, no inter-voxel dependencies, trivially parallelizable.

### Why Cone Beam is Harder

Cone beam geometry involves:
- Non-uniform ray spacing (perspective projection)
- More complex interpolation (trilinear vs bilinear)
- Higher memory bandwidth requirements
- More floating-point operations per voxel

### FPGA Suitability Notes

| Operation | Memory Pattern | Compute Pattern | FPGA Fit |
|-----------|----------------|-----------------|----------|
| forward_project | Streaming reads, scattered writes | Regular arithmetic | Good |
| back_project | Scattered reads, streaming writes | Regular arithmetic | Good |
| median_filter3d | Sliding window (local) | Sort/compare | Excellent |
| hessian_diagonal | Similar to forward_project | Regular arithmetic | Good |

## Raw Data Reference

See `mbirjax_profile_20260206_084448.json` for complete timing measurements.
