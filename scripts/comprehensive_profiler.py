#!/usr/bin/env python3
"""
MBIRJAX Comprehensive Profiler

Profiles all MBIRJAX operations across multiple volume sizes.
Outputs raw timing data in JSON format for external analysis.

Usage:
    python comprehensive_profiler.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp

# Block GUI dependencies
sys.modules['easygui'] = type(sys)('easygui')

from mbirjax import ParallelBeamModel, ConeBeamModel
from mbirjax import gen_pixel_partition
import mbirjax

OUTPUT_DIR = Path('/output')

# Fixed volume sizes for complexity analysis
VOLUME_SIZES = [32, 64, 128, 256]
RUNS_PER_SIZE = 3


def create_phantom(size: int) -> jnp.ndarray:
    """Create a test phantom volume."""
    phantom = np.zeros((size, size, size), dtype=np.float32)
    z, y, x = np.ogrid[:size, :size, :size]
    z = (z - size/2) / (size/2)
    y = (y - size/2) / (size/2)
    x = (x - size/2) / (size/2)
    mask = (x**2/0.7**2 + y**2/0.9**2 + z**2/0.85**2) <= 1
    phantom[mask] = 1.0
    mask = (x**2/0.5**2 + y**2/0.7**2 + z**2/0.6**2) <= 1
    phantom[mask] = 0.8
    return jnp.array(phantom)


def time_operation(func, *args, **kwargs):
    """Time a single operation, ensuring JAX synchronization."""
    t0 = time.time()
    result = func(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, tuple) and len(result) > 0 and hasattr(result[0], 'block_until_ready'):
        result[0].block_until_ready()
    return time.time() - t0, result


def profile_parallel_beam(phantom, vol_size, num_views):
    """Profile all ParallelBeamModel operations."""
    timings = {}

    angles = jnp.array(np.linspace(0, np.pi, num_views, endpoint=False), dtype=jnp.float32)
    model = ParallelBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles,
    )
    model.set_params(recon_shape=(vol_size, vol_size, vol_size))
    partitions = gen_pixel_partition((vol_size, vol_size, vol_size), 4)

    # Forward projection
    t, sinogram = time_operation(model.forward_project, phantom)
    timings['parallel_forward_project'] = t

    # Back projection
    t, _ = time_operation(model.back_project, sinogram)
    timings['parallel_back_project'] = t

    # Sparse forward projection
    try:
        pixel_indices = partitions[0]
        voxel_values = phantom.reshape(vol_size * vol_size, vol_size)[pixel_indices]
        t, _ = time_operation(model.sparse_forward_project, voxel_values, pixel_indices)
        timings['parallel_sparse_forward_project'] = t
    except Exception:
        pass

    # Sparse back projection
    try:
        pixel_indices = partitions[0]
        t, _ = time_operation(model.sparse_back_project, sinogram, pixel_indices)
        timings['parallel_sparse_back_project'] = t
    except Exception:
        pass

    # Hessian diagonal
    t, _ = time_operation(model.compute_hessian_diagonal)
    timings['parallel_hessian_diagonal'] = t

    # Direct filter
    try:
        t, _ = time_operation(model.direct_filter, sinogram)
        timings['parallel_direct_filter'] = t
    except Exception:
        pass

    # Add noise for reconstruction
    key = jax.random.PRNGKey(42)
    sinogram_noisy = sinogram + 0.01 * jnp.max(sinogram) * jax.random.normal(key, sinogram.shape)
    sinogram_noisy.block_until_ready()

    # MBIR reconstruction (1 iteration)
    t, _ = time_operation(model.recon, sinogram_noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
    timings['parallel_mbir_recon'] = t

    # Proximal map (1 iteration, exercises VCD + QGGMRF internally)
    try:
        t, _ = time_operation(model.prox_map, phantom, sinogram_noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
        timings['parallel_prox_map'] = t
    except Exception:
        pass

    # FBP reconstruction
    t, _ = time_operation(model.fbp_recon, sinogram_noisy)
    timings['parallel_fbp_recon'] = t

    # FBP filter
    t, _ = time_operation(model.fbp_filter, sinogram_noisy)
    timings['parallel_fbp_filter'] = t

    # Direct reconstruction
    t, _ = time_operation(model.direct_recon, sinogram_noisy)
    timings['parallel_direct_recon'] = t

    # Weight generation
    try:
        t, _ = time_operation(mbirjax.gen_weights, sinogram, 'transmission')
        timings['gen_weights_transmission'] = t
    except Exception:
        pass

    try:
        t, _ = time_operation(mbirjax.gen_weights_mar, model, sinogram)
        timings['parallel_gen_weights_mar'] = t
    except Exception:
        pass

    return timings


def profile_cone_beam(phantom, vol_size, num_views):
    """Profile all ConeBeamModel operations."""
    timings = {}

    angles = jnp.array(np.linspace(0, 2*np.pi, num_views, endpoint=False), dtype=jnp.float32)
    model = ConeBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles,
        source_detector_dist=4.0 * vol_size,
        source_iso_dist=2.0 * vol_size,
    )
    model.set_params(recon_shape=(vol_size, vol_size, vol_size))
    partitions = gen_pixel_partition((vol_size, vol_size, vol_size), 4)

    # Forward projection
    t, sinogram = time_operation(model.forward_project, phantom)
    timings['cone_forward_project'] = t

    # Back projection
    t, _ = time_operation(model.back_project, sinogram)
    timings['cone_back_project'] = t

    # Sparse forward projection
    try:
        pixel_indices = partitions[0]
        voxel_values = phantom.reshape(vol_size * vol_size, vol_size)[pixel_indices]
        t, _ = time_operation(model.sparse_forward_project, voxel_values, pixel_indices)
        timings['cone_sparse_forward_project'] = t
    except Exception:
        pass

    # Sparse back projection
    try:
        pixel_indices = partitions[0]
        t, _ = time_operation(model.sparse_back_project, sinogram, pixel_indices)
        timings['cone_sparse_back_project'] = t
    except Exception:
        pass

    # Hessian diagonal
    t, _ = time_operation(model.compute_hessian_diagonal)
    timings['cone_hessian_diagonal'] = t

    # Direct filter
    try:
        t, _ = time_operation(model.direct_filter, sinogram)
        timings['cone_direct_filter'] = t
    except Exception:
        pass

    # Add noise for reconstruction
    key = jax.random.PRNGKey(42)
    sinogram_noisy = sinogram + 0.01 * jnp.max(sinogram) * jax.random.normal(key, sinogram.shape)
    sinogram_noisy.block_until_ready()

    # MBIR reconstruction (1 iteration)
    t, _ = time_operation(model.recon, sinogram_noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
    timings['cone_mbir_recon'] = t

    # Proximal map (1 iteration, exercises VCD + QGGMRF internally)
    try:
        t, _ = time_operation(model.prox_map, phantom, sinogram_noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
        timings['cone_prox_map'] = t
    except Exception:
        pass

    # FDK reconstruction
    t, _ = time_operation(model.fdk_recon, sinogram_noisy)
    timings['cone_fdk_recon'] = t

    # FDK filter
    t, _ = time_operation(model.fdk_filter, sinogram_noisy)
    timings['cone_fdk_filter'] = t

    # Direct reconstruction
    try:
        t, _ = time_operation(model.direct_recon, sinogram_noisy)
        timings['cone_direct_recon'] = t
    except Exception:
        pass

    # Weight generation (MAR)
    try:
        t, _ = time_operation(mbirjax.gen_weights_mar, model, sinogram)
        timings['cone_gen_weights_mar'] = t
    except Exception:
        pass

    return timings


def profile_denoiser(phantom, vol_size):
    """Profile QGGMRFDenoiser (exercises QGGMRF regularization kernels)."""
    timings = {}
    key = jax.random.PRNGKey(42)
    noisy = phantom + 0.05 * jnp.max(phantom) * jax.random.normal(key, phantom.shape)
    noisy.block_until_ready()

    try:
        from mbirjax import QGGMRFDenoiser
        denoiser = QGGMRFDenoiser(noisy.shape)
        t, _ = time_operation(denoiser.denoise, noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
        timings['qggmrf_denoise'] = t
    except Exception:
        pass

    return timings


def profile_utilities(phantom, vol_size):
    """Profile utility operations."""
    timings = {}
    key = jax.random.PRNGKey(42)
    recon_shape = (vol_size, vol_size, vol_size)

    # Median filter 3D
    try:
        from mbirjax import median_filter3d
        noisy = phantom + 0.1 * jax.random.normal(key, phantom.shape)
        noisy.block_until_ready()
        t, _ = time_operation(median_filter3d, noisy)
        timings['median_filter3d'] = t
    except Exception:
        pass

    # Pixel partition generation variants
    t0 = time.time()
    gen_pixel_partition(recon_shape, 4)
    timings['gen_pixel_partition'] = time.time() - t0

    try:
        from mbirjax import gen_pixel_partition_blue_noise
        t0 = time.time()
        gen_pixel_partition_blue_noise(recon_shape, 4)
        timings['gen_pixel_partition_blue_noise'] = time.time() - t0
    except Exception:
        pass

    try:
        from mbirjax import gen_pixel_partition_grid
        t0 = time.time()
        gen_pixel_partition_grid(recon_shape, 4)
        timings['gen_pixel_partition_grid'] = time.time() - t0
    except Exception:
        pass

    return timings


def run_profiling_pass(vol_size: int):
    """Run one complete profiling pass for a given volume size."""
    num_views = vol_size // 2
    phantom = create_phantom(vol_size)

    timings = {}
    timings.update(profile_parallel_beam(phantom, vol_size, num_views))
    timings.update(profile_cone_beam(phantom, vol_size, num_views))
    timings.update(profile_denoiser(phantom, vol_size))
    timings.update(profile_utilities(phantom, vol_size))

    return timings


def main():
    print("\n" + "=" * 60)
    print("MBIRJAX COMPREHENSIVE PROFILER")
    print("=" * 60)
    print(f"Backend: {jax.default_backend()}")
    print(f"Volume sizes: {VOLUME_SIZES}")
    print(f"Runs per size: {RUNS_PER_SIZE}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'backend': str(jax.default_backend()),
            'devices': [str(d) for d in jax.devices()],
            'jax_version': jax.__version__,
            'mbirjax_version': getattr(mbirjax, '__version__', 'unknown'),
        },
        'volume_sizes': VOLUME_SIZES,
        'runs_per_size': RUNS_PER_SIZE,
        'measurements': []
    }

    for vol_size in VOLUME_SIZES:
        print(f"\n{'=' * 60}")
        print(f"Volume size: {vol_size}\u00b3 (views: {vol_size // 2})")
        print("=" * 60)

        for run in range(RUNS_PER_SIZE):
            print(f"\n  Run {run + 1}/{RUNS_PER_SIZE}")
            timings = run_profiling_pass(vol_size)

            for operation, elapsed in timings.items():
                results['measurements'].append({
                    'operation': operation,
                    'volume_size': vol_size,
                    'run': run + 1,
                    'time': elapsed
                })
                print(f"    {operation}: {elapsed:.4f}s")

    # Save raw JSON results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = OUTPUT_DIR / f"mbirjax_profile_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved: {json_file.name}")


if __name__ == '__main__':
    main()
