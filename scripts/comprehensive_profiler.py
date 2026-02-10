#!/usr/bin/env python3
"""
MBIRJAX Comprehensive GPU Profiler

Phase 1: XLA-level profiling for FPGA candidate discovery.
Captures JAX profiler traces (viewable in TensorBoard/XProf),
XLA cost analysis, and HLO computation graphs per operation.

For each volume size, three runs are performed:
  - Run 1: JIT warmup (compilation happens here)
  - Run 2: Traced via jax.profiler.trace (captures XLA execution timeline)
  - Run 3: Timing only (for comparison with traced run)

Output:
  jax_traces/<timestamp>/vol<N>/   - TensorBoard/XProf trace per volume size
  hlo_dumps/<timestamp>/           - HLO text per operation per volume size
  mbirjax_profile_<timestamp>.json - Wall-clock timing + XLA cost analysis

View traces:
  tensorboard --logdir=/output/jax_traces/<timestamp>

Usage:
    python comprehensive_profiler.py
"""

import sys
import time
import json
import contextlib
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
TRACE_RUN = 1  # 0-indexed; capture traces on second run (after JIT warmup, with command buffers disabled for kernel visibility)


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


_step_num = [0]


def time_operation(func, *args, **kwargs):
    """Time a single operation, ensuring JAX synchronization."""
    t0 = time.time()
    result = func(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, tuple) and len(result) > 0 and hasattr(result[0], 'block_until_ready'):
        result[0].block_until_ready()
    return time.time() - t0, result


@contextlib.contextmanager
def step(name):
    """Wrap an operation in a StepTraceAnnotation for XProf step analysis."""
    with jax.profiler.StepTraceAnnotation(name, step_num=_step_num[0]):
        _step_num[0] += 1
        yield


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
    with step('parallel_forward_project'):
        t, sinogram = time_operation(model.forward_project, phantom)
    timings['parallel_forward_project'] = t

    # Back projection
    with step('parallel_back_project'):
        t, _ = time_operation(model.back_project, sinogram)
    timings['parallel_back_project'] = t

    # Sparse forward projection
    try:
        pixel_indices = partitions[0]
        voxel_values = phantom.reshape(vol_size * vol_size, vol_size)[pixel_indices]
        with step('parallel_sparse_forward_project'):
            t, _ = time_operation(model.sparse_forward_project, voxel_values, pixel_indices)
        timings['parallel_sparse_forward_project'] = t
    except Exception:
        pass

    # Sparse back projection
    try:
        pixel_indices = partitions[0]
        with step('parallel_sparse_back_project'):
            t, _ = time_operation(model.sparse_back_project, sinogram, pixel_indices)
        timings['parallel_sparse_back_project'] = t
    except Exception:
        pass

    # Hessian diagonal
    with step('parallel_hessian_diagonal'):
        t, _ = time_operation(model.compute_hessian_diagonal)
    timings['parallel_hessian_diagonal'] = t

    # Direct filter
    try:
        with step('parallel_direct_filter'):
            t, _ = time_operation(model.direct_filter, sinogram)
        timings['parallel_direct_filter'] = t
    except Exception:
        pass

    # Add noise for reconstruction
    key = jax.random.PRNGKey(42)
    sinogram_noisy = sinogram + 0.01 * jnp.max(sinogram) * jax.random.normal(key, sinogram.shape)
    sinogram_noisy.block_until_ready()

    # MBIR reconstruction (1 iteration)
    with step('parallel_mbir_recon'):
        t, _ = time_operation(model.recon, sinogram_noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
    timings['parallel_mbir_recon'] = t

    # Proximal map (1 iteration, exercises VCD + QGGMRF internally)
    try:
        with step('parallel_prox_map'):
            t, _ = time_operation(model.prox_map, phantom, sinogram_noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
        timings['parallel_prox_map'] = t
    except Exception:
        pass

    # FBP reconstruction
    with step('parallel_fbp_recon'):
        t, _ = time_operation(model.fbp_recon, sinogram_noisy)
    timings['parallel_fbp_recon'] = t

    # FBP filter
    with step('parallel_fbp_filter'):
        t, _ = time_operation(model.fbp_filter, sinogram_noisy)
    timings['parallel_fbp_filter'] = t

    # Direct reconstruction
    with step('parallel_direct_recon'):
        t, _ = time_operation(model.direct_recon, sinogram_noisy)
    timings['parallel_direct_recon'] = t

    # Weight generation
    try:
        with step('gen_weights_transmission'):
            t, _ = time_operation(mbirjax.gen_weights, sinogram, 'transmission')
        timings['gen_weights_transmission'] = t
    except Exception:
        pass

    try:
        with step('parallel_gen_weights_mar'):
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
    with step('cone_forward_project'):
        t, sinogram = time_operation(model.forward_project, phantom)
    timings['cone_forward_project'] = t

    # Back projection
    with step('cone_back_project'):
        t, _ = time_operation(model.back_project, sinogram)
    timings['cone_back_project'] = t

    # Sparse forward projection
    try:
        pixel_indices = partitions[0]
        voxel_values = phantom.reshape(vol_size * vol_size, vol_size)[pixel_indices]
        with step('cone_sparse_forward_project'):
            t, _ = time_operation(model.sparse_forward_project, voxel_values, pixel_indices)
        timings['cone_sparse_forward_project'] = t
    except Exception:
        pass

    # Sparse back projection
    try:
        pixel_indices = partitions[0]
        with step('cone_sparse_back_project'):
            t, _ = time_operation(model.sparse_back_project, sinogram, pixel_indices)
        timings['cone_sparse_back_project'] = t
    except Exception:
        pass

    # Hessian diagonal
    with step('cone_hessian_diagonal'):
        t, _ = time_operation(model.compute_hessian_diagonal)
    timings['cone_hessian_diagonal'] = t

    # Direct filter
    try:
        with step('cone_direct_filter'):
            t, _ = time_operation(model.direct_filter, sinogram)
        timings['cone_direct_filter'] = t
    except Exception:
        pass

    # Add noise for reconstruction
    key = jax.random.PRNGKey(42)
    sinogram_noisy = sinogram + 0.01 * jnp.max(sinogram) * jax.random.normal(key, sinogram.shape)
    sinogram_noisy.block_until_ready()

    # MBIR reconstruction (1 iteration)
    with step('cone_mbir_recon'):
        t, _ = time_operation(model.recon, sinogram_noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
    timings['cone_mbir_recon'] = t

    # Proximal map (1 iteration, exercises VCD + QGGMRF internally)
    try:
        with step('cone_prox_map'):
            t, _ = time_operation(model.prox_map, phantom, sinogram_noisy, max_iterations=1, stop_threshold_change_pct=0, print_logs=False)
        timings['cone_prox_map'] = t
    except Exception:
        pass

    # FDK reconstruction
    with step('cone_fdk_recon'):
        t, _ = time_operation(model.fdk_recon, sinogram_noisy)
    timings['cone_fdk_recon'] = t

    # FDK filter
    with step('cone_fdk_filter'):
        t, _ = time_operation(model.fdk_filter, sinogram_noisy)
    timings['cone_fdk_filter'] = t

    # Direct reconstruction
    try:
        with step('cone_direct_recon'):
            t, _ = time_operation(model.direct_recon, sinogram_noisy)
        timings['cone_direct_recon'] = t
    except Exception:
        pass

    # Weight generation (MAR)
    try:
        with step('cone_gen_weights_mar'):
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
        with step('qggmrf_denoise'):
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
        with step('median_filter3d'):
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


def run_profiling_pass(vol_size):
    """Run one complete profiling pass for a given volume size.

    Each operation is annotated with StepTraceAnnotation so XProf
    can attribute GPU kernels to individual operations.
    """
    _step_num[0] = 0
    num_views = vol_size // 2
    phantom = create_phantom(vol_size)

    timings = {}
    timings.update(profile_parallel_beam(phantom, vol_size, num_views))
    timings.update(profile_cone_beam(phantom, vol_size, num_views))
    timings.update(profile_denoiser(phantom, vol_size))
    timings.update(profile_utilities(phantom, vol_size))

    return timings


def collect_cost_analysis(vol_size):
    """Collect XLA cost analysis (FLOPs, bytes accessed) for key operations.

    Attempts jax.jit(func).lower(*args).compile().cost_analysis() for each
    operation. This may fail for internally-jitted methods; failures are
    recorded as errors rather than raised.
    """
    costs = {}
    phantom = create_phantom(vol_size)
    num_views = vol_size // 2

    # --- Parallel beam operations ---
    angles = jnp.array(np.linspace(0, np.pi, num_views, endpoint=False), dtype=jnp.float32)
    par_model = ParallelBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles,
    )
    par_model.set_params(recon_shape=(vol_size, vol_size, vol_size))

    sinogram_par = par_model.forward_project(phantom)
    sinogram_par.block_until_ready()

    par_ops = [
        ('parallel_forward_project', par_model.forward_project, (phantom,)),
        ('parallel_back_project', par_model.back_project, (sinogram_par,)),
        ('parallel_hessian_diagonal', par_model.compute_hessian_diagonal, ()),
        ('parallel_fbp_recon', par_model.fbp_recon, (sinogram_par,)),
        ('parallel_fbp_filter', par_model.fbp_filter, (sinogram_par,)),
    ]

    # --- Cone beam operations ---
    angles_cone = jnp.array(np.linspace(0, 2 * np.pi, num_views, endpoint=False), dtype=jnp.float32)
    cone_model = ConeBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles_cone,
        source_detector_dist=4.0 * vol_size,
        source_iso_dist=2.0 * vol_size,
    )
    cone_model.set_params(recon_shape=(vol_size, vol_size, vol_size))

    sinogram_cone = cone_model.forward_project(phantom)
    sinogram_cone.block_until_ready()

    cone_ops = [
        ('cone_forward_project', cone_model.forward_project, (phantom,)),
        ('cone_back_project', cone_model.back_project, (sinogram_cone,)),
        ('cone_hessian_diagonal', cone_model.compute_hessian_diagonal, ()),
        ('cone_fdk_recon', cone_model.fdk_recon, (sinogram_cone,)),
        ('cone_fdk_filter', cone_model.fdk_filter, (sinogram_cone,)),
    ]

    all_ops = par_ops + cone_ops

    for name, func, args in all_ops:
        try:
            lowered = jax.jit(func).lower(*args)
            compiled = lowered.compile()
            cost = compiled.cost_analysis()
            if cost and len(cost) > 0:
                entry = {}
                for k, v in cost[0].items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        entry[k] = float(v)
                    else:
                        entry[k] = str(v)
                costs[name] = entry
                print(f"    {name}: {entry.get('flops', '?')} flops, "
                      f"{entry.get('bytes accessed', '?')} bytes")
            else:
                costs[name] = {'note': 'cost_analysis returned empty'}
        except Exception as e:
            costs[name] = {'error': str(e)}
            print(f"    {name}: cost analysis failed ({e})")

    return costs


def dump_hlo(vol_size, hlo_dir):
    """Dump HLO text (StableHLO) for key operations.

    Uses jax.jit(func).lower(*args).as_text() to get the XLA computation
    graph in human-readable form.
    """
    hlo_dir.mkdir(parents=True, exist_ok=True)
    phantom = create_phantom(vol_size)
    num_views = vol_size // 2
    dumped = 0

    # --- Parallel beam ---
    angles = jnp.array(np.linspace(0, np.pi, num_views, endpoint=False), dtype=jnp.float32)
    par_model = ParallelBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles,
    )
    par_model.set_params(recon_shape=(vol_size, vol_size, vol_size))
    sinogram_par = par_model.forward_project(phantom)
    sinogram_par.block_until_ready()

    # --- Cone beam ---
    angles_cone = jnp.array(np.linspace(0, 2 * np.pi, num_views, endpoint=False), dtype=jnp.float32)
    cone_model = ConeBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles_cone,
        source_detector_dist=4.0 * vol_size,
        source_iso_dist=2.0 * vol_size,
    )
    cone_model.set_params(recon_shape=(vol_size, vol_size, vol_size))
    sinogram_cone = cone_model.forward_project(phantom)
    sinogram_cone.block_until_ready()

    all_ops = [
        ('parallel_forward_project', par_model.forward_project, (phantom,)),
        ('parallel_back_project', par_model.back_project, (sinogram_par,)),
        ('parallel_hessian_diagonal', par_model.compute_hessian_diagonal, ()),
        ('parallel_fbp_recon', par_model.fbp_recon, (sinogram_par,)),
        ('cone_forward_project', cone_model.forward_project, (phantom,)),
        ('cone_back_project', cone_model.back_project, (sinogram_cone,)),
        ('cone_hessian_diagonal', cone_model.compute_hessian_diagonal, ()),
        ('cone_fdk_recon', cone_model.fdk_recon, (sinogram_cone,)),
    ]

    for name, func, args in all_ops:
        try:
            lowered = jax.jit(func).lower(*args)
            hlo_text = lowered.as_text()
            out_path = hlo_dir / f'{name}_vol{vol_size}.txt'
            out_path.write_text(hlo_text)
            dumped += 1
        except Exception:
            pass

    return dumped


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trace_base = OUTPUT_DIR / 'jax_traces' / timestamp
    hlo_dir = OUTPUT_DIR / 'hlo_dumps' / timestamp

    print("\n" + "=" * 60)
    print("MBIRJAX GPU PROFILER — Phase 1 (XLA-level)")
    print("=" * 60)
    print(f"Backend:      {jax.default_backend()}")
    print(f"Devices:      {jax.devices()}")
    print(f"JAX version:  {jax.__version__}")
    print(f"Volume sizes: {VOLUME_SIZES}")
    print(f"Runs/size:    {RUNS_PER_SIZE} (run 1 warmup, run 2 traced, run 3 timing)")

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
        'measurements': [],
        'cost_analysis': {},
    }

    for vol_size in VOLUME_SIZES:
        print(f"\n{'=' * 60}")
        print(f"Volume size: {vol_size}\u00b3 (views: {vol_size // 2})")
        print("=" * 60)

        for run in range(RUNS_PER_SIZE):
            # Capture XLA trace on Run 2 (after JIT warmup)
            if run == TRACE_RUN:
                trace_dir = trace_base / f'vol{vol_size}'
                trace_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n  Run {run + 1}/{RUNS_PER_SIZE} [TRACED -> {trace_dir.relative_to(OUTPUT_DIR)}]")
            else:
                trace_dir = None
                label = "warmup" if run == 0 else ""
                suffix = f" [{label}]" if label else ""
                print(f"\n  Run {run + 1}/{RUNS_PER_SIZE}{suffix}")

            ctx = jax.profiler.trace(str(trace_dir)) if trace_dir else contextlib.nullcontext()
            with ctx:
                timings = run_profiling_pass(vol_size)

            for operation, elapsed in timings.items():
                results['measurements'].append({
                    'operation': operation,
                    'volume_size': vol_size,
                    'run': run + 1,
                    'time': elapsed,
                    'traced': run == TRACE_RUN,
                })
                print(f"    {operation}: {elapsed:.4f}s")

        # Cost analysis (FLOPs, bytes accessed, arithmetic intensity)
        print(f"\n  Cost analysis (vol {vol_size}\u00b3):")
        costs = collect_cost_analysis(vol_size)
        results['cost_analysis'][str(vol_size)] = costs

        # HLO computation graph dumps
        print(f"\n  HLO dumps (vol {vol_size}\u00b3):")
        n_dumped = dump_hlo(vol_size, hlo_dir)
        print(f"    {n_dumped} HLO graphs saved to {hlo_dir.relative_to(OUTPUT_DIR)}/")

    # Save JSON results (timing + cost analysis)
    json_file = OUTPUT_DIR / f"mbirjax_profile_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("PROFILING COMPLETE")
    print("=" * 60)
    print(f"\nOutput:")
    print(f"  Timing + cost analysis: {json_file.name}")
    print(f"  XLA traces:            jax_traces/{timestamp}/")
    print(f"  HLO dumps:             hlo_dumps/{timestamp}/")
    print(f"\nView traces in TensorBoard:")
    print(f"  tensorboard --logdir=/output/jax_traces/{timestamp}")
    print(f"\nXProf tools to use:")
    print(f"  - Trace Viewer:     GPU compute vs memory transfer timeline")
    print(f"  - Roofline Analysis: identify memory-bound ops (FPGA candidates)")
    print(f"  - HLO Op Stats:     per-operation time and arithmetic intensity")
    print(f"  - Memory Viewer:    buffer sizes for FPGA on-chip memory planning")


if __name__ == '__main__':
    main()
