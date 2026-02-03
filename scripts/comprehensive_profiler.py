#!/usr/bin/env python3
"""
MBIRJAX Comprehensive Profiler for FPGA Acceleration Analysis

Profiles ALL MBIRJAX operations across multiple volume sizes.
Outputs JSON suitable for AI analysis of FPGA implementation candidates.

Usage:
  python comprehensive_profiler.py [--preset small|medium|large]
"""

import sys
import time
import json
import cProfile
import pstats
from pathlib import Path
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp

# Block GUI dependencies
sys.modules['easygui'] = type(sys)('easygui')

from mbirjax import ParallelBeamModel, ConeBeamModel
import mbirjax

OUTPUT_DIR = Path('/output')


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


def run_all_operations(vol_size: int, num_views: int, num_iters: int):
    """Run all MBIRJAX operations for a given volume size."""
    results = {}

    print(f"\n{'='*60}")
    print(f"Volume: {vol_size}³, Views: {num_views}, Iterations: {num_iters}")
    print(f"{'='*60}")

    phantom = create_phantom(vol_size)

    # ==================== PARALLEL BEAM ====================
    print("\n[Parallel Beam Model]")
    angles = jnp.array(np.linspace(0, np.pi, num_views, endpoint=False), dtype=jnp.float32)
    model = ParallelBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles,
    )
    model.set_params(recon_shape=(vol_size, vol_size, vol_size))

    # Forward projection
    t0 = time.time()
    sinogram = model.forward_project(phantom)
    sinogram.block_until_ready()
    results[f'parallel_forward_{vol_size}'] = time.time() - t0
    print(f"  forward_project: {results[f'parallel_forward_{vol_size}']:.3f}s")

    # Back projection
    t0 = time.time()
    back = model.back_project(sinogram)
    back.block_until_ready()
    results[f'parallel_back_{vol_size}'] = time.time() - t0
    print(f"  back_project: {results[f'parallel_back_{vol_size}']:.3f}s")

    # Hessian diagonal
    t0 = time.time()
    hess = model.compute_hessian_diagonal()
    hess.block_until_ready()
    results[f'parallel_hessian_{vol_size}'] = time.time() - t0
    print(f"  compute_hessian_diagonal: {results[f'parallel_hessian_{vol_size}']:.3f}s")

    # Add noise for reconstruction
    key = jax.random.PRNGKey(42)
    sinogram_noisy = sinogram + 0.01 * jnp.max(sinogram) * jax.random.normal(key, sinogram.shape)
    sinogram_noisy.block_until_ready()

    # MBIR reconstruction
    t0 = time.time()
    recon_result = model.recon(sinogram_noisy, max_iterations=num_iters, stop_threshold_change_pct=0, print_logs=False)
    recon = recon_result[0] if isinstance(recon_result, tuple) else recon_result
    recon.block_until_ready()
    results[f'parallel_mbir_{vol_size}'] = time.time() - t0
    print(f"  mbir_recon: {results[f'parallel_mbir_{vol_size}']:.3f}s")

    # FBP reconstruction
    t0 = time.time()
    fbp = model.fbp_recon(sinogram_noisy)
    fbp.block_until_ready()
    results[f'parallel_fbp_{vol_size}'] = time.time() - t0
    print(f"  fbp_recon: {results[f'parallel_fbp_{vol_size}']:.3f}s")

    # FBP filter
    t0 = time.time()
    filtered = model.fbp_filter(sinogram_noisy)
    filtered.block_until_ready()
    results[f'parallel_fbp_filter_{vol_size}'] = time.time() - t0
    print(f"  fbp_filter: {results[f'parallel_fbp_filter_{vol_size}']:.3f}s")

    # Direct reconstruction
    t0 = time.time()
    direct = model.direct_recon(sinogram_noisy)
    direct.block_until_ready()
    results[f'parallel_direct_{vol_size}'] = time.time() - t0
    print(f"  direct_recon: {results[f'parallel_direct_{vol_size}']:.3f}s")

    # ==================== CONE BEAM ====================
    print("\n[Cone Beam Model]")
    angles_cone = jnp.array(np.linspace(0, 2*np.pi, num_views, endpoint=False), dtype=jnp.float32)
    cone_model = ConeBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles_cone,
        source_detector_dist=4.0 * vol_size,
        source_iso_dist=2.0 * vol_size,
    )
    cone_model.set_params(recon_shape=(vol_size, vol_size, vol_size))

    # Forward projection
    t0 = time.time()
    sino_cone = cone_model.forward_project(phantom)
    sino_cone.block_until_ready()
    results[f'cone_forward_{vol_size}'] = time.time() - t0
    print(f"  forward_project: {results[f'cone_forward_{vol_size}']:.3f}s")

    # Back projection
    t0 = time.time()
    back_cone = cone_model.back_project(sino_cone)
    back_cone.block_until_ready()
    results[f'cone_back_{vol_size}'] = time.time() - t0
    print(f"  back_project: {results[f'cone_back_{vol_size}']:.3f}s")

    # Hessian diagonal
    t0 = time.time()
    hess_cone = cone_model.compute_hessian_diagonal()
    hess_cone.block_until_ready()
    results[f'cone_hessian_{vol_size}'] = time.time() - t0
    print(f"  compute_hessian_diagonal: {results[f'cone_hessian_{vol_size}']:.3f}s")

    # Add noise
    sino_cone_noisy = sino_cone + 0.01 * jnp.max(sino_cone) * jax.random.normal(key, sino_cone.shape)
    sino_cone_noisy.block_until_ready()

    # MBIR reconstruction
    t0 = time.time()
    recon_cone_result = cone_model.recon(sino_cone_noisy, max_iterations=num_iters, stop_threshold_change_pct=0, print_logs=False)
    recon_cone = recon_cone_result[0] if isinstance(recon_cone_result, tuple) else recon_cone_result
    recon_cone.block_until_ready()
    results[f'cone_mbir_{vol_size}'] = time.time() - t0
    print(f"  mbir_recon: {results[f'cone_mbir_{vol_size}']:.3f}s")

    # FDK reconstruction
    t0 = time.time()
    fdk = cone_model.fdk_recon(sino_cone_noisy)
    fdk.block_until_ready()
    results[f'cone_fdk_{vol_size}'] = time.time() - t0
    print(f"  fdk_recon: {results[f'cone_fdk_{vol_size}']:.3f}s")

    # FDK filter
    t0 = time.time()
    fdk_filt = cone_model.fdk_filter(sino_cone_noisy)
    fdk_filt.block_until_ready()
    results[f'cone_fdk_filter_{vol_size}'] = time.time() - t0
    print(f"  fdk_filter: {results[f'cone_fdk_filter_{vol_size}']:.3f}s")

    # ==================== ADDITIONAL OPERATIONS ====================
    print("\n[Additional Operations]")

    # Median filter 3D
    try:
        from mbirjax import median_filter3d
        noisy = phantom + 0.1 * jax.random.normal(key, phantom.shape)
        t0 = time.time()
        med = median_filter3d(noisy)
        med.block_until_ready()
        results[f'median_filter3d_{vol_size}'] = time.time() - t0
        print(f"  median_filter3d: {results[f'median_filter3d_{vol_size}']:.3f}s")
    except Exception as e:
        print(f"  median_filter3d: skipped ({e})")

    # Pixel partition generation
    try:
        from mbirjax import gen_pixel_partition
        t0 = time.time()
        partition = gen_pixel_partition((vol_size, vol_size, vol_size), 4)
        results[f'gen_pixel_partition_{vol_size}'] = time.time() - t0
        print(f"  gen_pixel_partition: {results[f'gen_pixel_partition_{vol_size}']:.3f}s")
    except Exception as e:
        print(f"  gen_pixel_partition: skipped ({e})")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MBIRJAX Comprehensive Profiler')
    parser.add_argument('--preset', choices=['small', 'medium', 'large'], default='small',
                       help='Profile preset: small (64,128), medium (64,128,256), large (128,256,512)')
    args = parser.parse_args()

    # Define volume sizes for each preset
    presets = {
        'small': [(64, 36, 2), (128, 72, 2)],
        'medium': [(64, 36, 2), (128, 72, 3), (256, 180, 3)],
        'large': [(128, 72, 3), (256, 180, 5), (512, 360, 5)],
    }

    configs = presets[args.preset]

    print("\n" + "="*60)
    print("MBIRJAX COMPREHENSIVE PROFILER")
    print("="*60)
    print(f"Preset: {args.preset}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Configurations: {configs}")

    # Start profiling
    profiler = cProfile.Profile()
    all_results = {
        'preset': args.preset,
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'backend': str(jax.default_backend()),
            'jax_version': jax.__version__,
            'mbirjax_version': getattr(mbirjax, '__version__', 'unknown'),
        },
        'configurations': [{'volume_size': v, 'num_views': n, 'num_iters': i} for v, n, i in configs],
        'timings': {},
        'by_operation': {},
        'by_geometry': {'parallel_beam': {}, 'cone_beam': {}},
        'scaling_analysis': {},
    }

    profiler.enable()

    for vol_size, num_views, num_iters in configs:
        results = run_all_operations(vol_size, num_views, num_iters)
        all_results['timings'].update(results)

    profiler.disable()

    # Save profile
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    profile_file = OUTPUT_DIR / f"mbirjax_profile_{args.preset}_{timestamp}.prof"
    profiler.dump_stats(str(profile_file))
    print(f"\n✓ Profile saved: {profile_file.name}")

    # Organize timings by operation type and geometry
    for key, time_val in all_results['timings'].items():
        parts = key.rsplit('_', 1)
        if len(parts) == 2:
            op_name, size = parts[0], int(parts[1])
            # By operation
            if op_name not in all_results['by_operation']:
                all_results['by_operation'][op_name] = {}
            all_results['by_operation'][op_name][size] = time_val
            # By geometry
            if op_name.startswith('parallel_'):
                op_short = op_name.replace('parallel_', '')
                if op_short not in all_results['by_geometry']['parallel_beam']:
                    all_results['by_geometry']['parallel_beam'][op_short] = {}
                all_results['by_geometry']['parallel_beam'][op_short][size] = time_val
            elif op_name.startswith('cone_'):
                op_short = op_name.replace('cone_', '')
                if op_short not in all_results['by_geometry']['cone_beam']:
                    all_results['by_geometry']['cone_beam'][op_short] = {}
                all_results['by_geometry']['cone_beam'][op_short][size] = time_val

    # Compute scaling analysis (time increase per volume size increase)
    sizes = sorted(set(int(k.rsplit('_', 1)[1]) for k in all_results['timings'].keys() if k.rsplit('_', 1)[1].isdigit()))
    for op_name, size_times in all_results['by_operation'].items():
        if len(size_times) >= 2:
            sorted_sizes = sorted(size_times.keys())
            scaling_factors = []
            for i in range(1, len(sorted_sizes)):
                s1, s2 = sorted_sizes[i-1], sorted_sizes[i]
                t1, t2 = size_times[s1], size_times[s2]
                if t1 > 0:
                    volume_ratio = (s2 / s1) ** 3
                    time_ratio = t2 / t1
                    scaling_factors.append({
                        'from_size': s1, 'to_size': s2,
                        'volume_ratio': round(volume_ratio, 2),
                        'time_ratio': round(time_ratio, 2),
                        'complexity_estimate': round(time_ratio / volume_ratio, 3) if volume_ratio > 0 else None
                    })
            all_results['scaling_analysis'][op_name] = scaling_factors

    # Add summary statistics
    all_results['summary'] = {
        'total_time': sum(all_results['timings'].values()),
        'slowest_operations': sorted(all_results['timings'].items(), key=lambda x: x[1], reverse=True)[:10],
        'operations_by_category': {
            'forward_projection': sum(v for k, v in all_results['timings'].items() if 'forward' in k),
            'back_projection': sum(v for k, v in all_results['timings'].items() if 'back' in k),
            'reconstruction': sum(v for k, v in all_results['timings'].items() if any(x in k for x in ['mbir', 'fbp', 'fdk', 'direct'])),
            'hessian': sum(v for k, v in all_results['timings'].items() if 'hessian' in k),
            'filtering': sum(v for k, v in all_results['timings'].items() if 'filter' in k),
        }
    }

    # Save JSON results
    json_file = OUTPUT_DIR / f"mbirjax_profile_{args.preset}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Results saved: {json_file.name}")

    # Print summary
    print("\n" + "="*60)
    print("TIMING SUMMARY (sorted by time)")
    print("="*60)
    sorted_timings = sorted(all_results['timings'].items(), key=lambda x: x[1], reverse=True)
    for name, duration in sorted_timings:
        print(f"  {name:<40} {duration:>8.3f}s")

    print("\n" + "="*60)
    print("FPGA CANDIDATES (top consumers)")
    print("="*60)
    for name, duration in sorted_timings[:10]:
        print(f"  • {name}: {duration:.3f}s")


if __name__ == '__main__':
    main()
