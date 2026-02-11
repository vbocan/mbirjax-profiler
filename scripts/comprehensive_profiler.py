#!/usr/bin/env python3
"""
MBIRJAX Comprehensive GPU Profiler

Phase 1: XLA-level profiling for FPGA candidate discovery.
Captures JAX profiler traces (viewable in TensorBoard/XProf),
XLA cost analysis, and HLO computation graphs per operation.

For each volume size, three runs are performed:
  - Run 1: JIT warmup (compilation happens here — HLO module protos captured)
  - Run 2: Post-warmup execution (kernel timing captured)
  - Run 3: Timing only (for comparison)
All runs are wrapped in a single jax.profiler.trace so XProf can correlate
HLO module protos (from JIT compilation) with kernel execution timing.

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
import argparse
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
TRACE_RUN = 1  # 0-indexed; execution-only trace (after JIT warmup)
TRACE_ALL = True  # Wrap ALL runs in a single trace so XProf captures HLO module protos during JIT compilation (run 0) AND kernel timing during execution (runs 1+)


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
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, tuple) and len(result) > 0 and hasattr(result[0], 'block_until_ready'):
        result[0].block_until_ready()
    return time.perf_counter() - t0, result


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
    t0 = time.perf_counter()
    gen_pixel_partition(recon_shape, 4)
    timings['gen_pixel_partition'] = time.perf_counter() - t0

    try:
        from mbirjax import gen_pixel_partition_blue_noise
        t0 = time.perf_counter()
        gen_pixel_partition_blue_noise(recon_shape, 4)
        timings['gen_pixel_partition_blue_noise'] = time.perf_counter() - t0
    except Exception:
        pass

    try:
        from mbirjax import gen_pixel_partition_grid
        t0 = time.perf_counter()
        gen_pixel_partition_grid(recon_shape, 4)
        timings['gen_pixel_partition_grid'] = time.perf_counter() - t0
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

    MBIRJAX's public methods (forward_project, back_project, etc.) contain
    Python control flow that cannot be re-traced by an outer jax.jit.
    Instead, we target the inner JIT-compiled kernels:
      - sparse_forward_project / sparse_back_project via model.projector_functions
      - A representative 1-D convolution kernel for filter operations
    Cost is reported per-partition (projection) or per-view (filter), with
    the multiplier included so total cost can be derived.
    """
    costs = {}
    phantom = create_phantom(vol_size)
    num_views = vol_size // 2
    num_partitions = 4
    recon_shape = (vol_size, vol_size, vol_size)

    def extract_cost(jit_fn, *args):
        """Lower, compile, and extract cost dict from a JIT'd function."""
        lowered = jit_fn.lower(*args)
        compiled = lowered.compile()
        cost = compiled.cost_analysis()
        # cost_analysis() returns a dict in newer JAX, a list in older versions
        if isinstance(cost, dict):
            cost_dict = cost
        elif isinstance(cost, (list, tuple)) and len(cost) > 0:
            cost_dict = cost[0]
        else:
            return None
        entry = {}
        for k, v in cost_dict.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                entry[k] = float(v)
            else:
                entry[k] = str(v)
        return entry if entry else None

    # --- Common inputs for projection kernels ---
    partitions = gen_pixel_partition(recon_shape, num_partitions)
    pixel_indices = partitions[0]
    voxel_values = phantom.reshape(vol_size * vol_size, vol_size)[pixel_indices]

    # --- Parallel beam ---
    angles = jnp.array(np.linspace(0, np.pi, num_views, endpoint=False), dtype=jnp.float32)
    par_model = ParallelBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles,
    )
    par_model.set_params(recon_shape=recon_shape)
    sinogram_par = par_model.forward_project(phantom)
    sinogram_par.block_until_ready()
    par_pf = par_model.projector_functions

    par_kernel_ops = [
        ('parallel_forward_project', 'sparse_forward_project',
         par_pf.sparse_forward_project, (voxel_values, pixel_indices)),
        ('parallel_back_project', 'sparse_back_project',
         par_pf.sparse_back_project, (sinogram_par, pixel_indices)),
        ('parallel_hessian_diagonal', 'sparse_back_project',
         par_pf.sparse_back_project, (sinogram_par, pixel_indices)),
    ]

    for name, kernel_name, jit_fn, args in par_kernel_ops:
        try:
            entry = extract_cost(jit_fn, *args)
            if entry:
                entry['kernel'] = kernel_name
                entry['num_partitions'] = num_partitions
                costs[name] = entry
                print(f"    {name}: {entry.get('flops', '?')} flops/partition, "
                      f"{entry.get('bytes accessed', '?')} bytes")
            else:
                costs[name] = {'note': 'cost_analysis returned empty'}
        except Exception as e:
            costs[name] = {'error': str(e)}
            print(f"    {name}: cost analysis failed ({e})")

    # --- Parallel beam filter kernel (representative 1-D convolution) ---
    det_cols = vol_size
    recon_filter_rep = jnp.ones(2 * det_cols - 1, dtype=jnp.float32)
    single_view = jnp.ones((vol_size, vol_size), dtype=jnp.float32)

    @jax.jit
    def par_filter_one_view(view):
        return jax.vmap(lambda row: jnp.convolve(row, recon_filter_rep, mode='same'))(view)

    try:
        entry = extract_cost(par_filter_one_view, single_view)
        if entry:
            entry['kernel'] = 'filter_one_view (representative)'
            entry['num_views'] = num_views
            costs['parallel_fbp_filter'] = entry
            print(f"    parallel_fbp_filter: {entry.get('flops', '?')} flops/view, "
                  f"{entry.get('bytes accessed', '?')} bytes")
        else:
            costs['parallel_fbp_filter'] = {'note': 'cost_analysis returned empty'}
    except Exception as e:
        costs['parallel_fbp_filter'] = {'error': str(e)}
        print(f"    parallel_fbp_filter: cost analysis failed ({e})")

    costs['parallel_fbp_recon'] = {
        'note': 'composite: parallel_fbp_filter + parallel_back_project',
    }

    # --- Cone beam ---
    angles_cone = jnp.array(np.linspace(0, 2 * np.pi, num_views, endpoint=False), dtype=jnp.float32)
    cone_model = ConeBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles_cone,
        source_detector_dist=4.0 * vol_size,
        source_iso_dist=2.0 * vol_size,
    )
    cone_model.set_params(recon_shape=recon_shape)
    sinogram_cone = cone_model.forward_project(phantom)
    sinogram_cone.block_until_ready()
    cone_pf = cone_model.projector_functions

    cone_kernel_ops = [
        ('cone_forward_project', 'sparse_forward_project',
         cone_pf.sparse_forward_project, (voxel_values, pixel_indices)),
        ('cone_back_project', 'sparse_back_project',
         cone_pf.sparse_back_project, (sinogram_cone, pixel_indices)),
        ('cone_hessian_diagonal', 'sparse_back_project',
         cone_pf.sparse_back_project, (sinogram_cone, pixel_indices)),
    ]

    for name, kernel_name, jit_fn, args in cone_kernel_ops:
        try:
            entry = extract_cost(jit_fn, *args)
            if entry:
                entry['kernel'] = kernel_name
                entry['num_partitions'] = num_partitions
                costs[name] = entry
                print(f"    {name}: {entry.get('flops', '?')} flops/partition, "
                      f"{entry.get('bytes accessed', '?')} bytes")
            else:
                costs[name] = {'note': 'cost_analysis returned empty'}
        except Exception as e:
            costs[name] = {'error': str(e)}
            print(f"    {name}: cost analysis failed ({e})")

    # --- Cone beam filter kernel (representative 1-D convolution) ---
    @jax.jit
    def cone_filter_one_view(view):
        return jax.vmap(lambda row: jnp.convolve(row, recon_filter_rep, mode='same'))(view)

    try:
        entry = extract_cost(cone_filter_one_view, single_view)
        if entry:
            entry['kernel'] = 'filter_one_view (representative)'
            entry['num_views'] = num_views
            costs['cone_fdk_filter'] = entry
            print(f"    cone_fdk_filter: {entry.get('flops', '?')} flops/view, "
                  f"{entry.get('bytes accessed', '?')} bytes")
        else:
            costs['cone_fdk_filter'] = {'note': 'cost_analysis returned empty'}
    except Exception as e:
        costs['cone_fdk_filter'] = {'error': str(e)}
        print(f"    cone_fdk_filter: cost analysis failed ({e})")

    costs['cone_fdk_recon'] = {
        'note': 'composite: cone_fdk_filter + cone_back_project',
    }

    return costs


def dump_hlo(vol_size, hlo_dir):
    """Dump HLO text (StableHLO) for inner JIT-compiled kernels.

    Targets the Projectors' sparse_forward/back_project functions and
    a representative filter convolution kernel, which are the actual
    GPU kernels suitable for FPGA analysis.
    """
    hlo_dir.mkdir(parents=True, exist_ok=True)
    phantom = create_phantom(vol_size)
    num_views = vol_size // 2
    num_partitions = 4
    recon_shape = (vol_size, vol_size, vol_size)
    dumped = 0

    partitions = gen_pixel_partition(recon_shape, num_partitions)
    pixel_indices = partitions[0]
    voxel_values = phantom.reshape(vol_size * vol_size, vol_size)[pixel_indices]

    # --- Parallel beam ---
    angles = jnp.array(np.linspace(0, np.pi, num_views, endpoint=False), dtype=jnp.float32)
    par_model = ParallelBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles,
    )
    par_model.set_params(recon_shape=recon_shape)
    sinogram_par = par_model.forward_project(phantom)
    sinogram_par.block_until_ready()
    par_pf = par_model.projector_functions

    # --- Cone beam ---
    angles_cone = jnp.array(np.linspace(0, 2 * np.pi, num_views, endpoint=False), dtype=jnp.float32)
    cone_model = ConeBeamModel(
        sinogram_shape=(num_views, vol_size, vol_size),
        angles=angles_cone,
        source_detector_dist=4.0 * vol_size,
        source_iso_dist=2.0 * vol_size,
    )
    cone_model.set_params(recon_shape=recon_shape)
    sinogram_cone = cone_model.forward_project(phantom)
    sinogram_cone.block_until_ready()
    cone_pf = cone_model.projector_functions

    # --- Projection / backprojection kernels ---
    all_ops = [
        ('sparse_forward_project_parallel', par_pf.sparse_forward_project,
         (voxel_values, pixel_indices)),
        ('sparse_back_project_parallel', par_pf.sparse_back_project,
         (sinogram_par, pixel_indices)),
        ('sparse_forward_project_cone', cone_pf.sparse_forward_project,
         (voxel_values, pixel_indices)),
        ('sparse_back_project_cone', cone_pf.sparse_back_project,
         (sinogram_cone, pixel_indices)),
    ]

    for name, jit_fn, args in all_ops:
        try:
            lowered = jit_fn.lower(*args)
            hlo_text = lowered.as_text()
            out_path = hlo_dir / f'{name}_vol{vol_size}.txt'
            out_path.write_text(hlo_text)
            dumped += 1
        except Exception:
            pass

    # --- Filter kernel ---
    det_cols = vol_size
    recon_filter = jnp.ones(2 * det_cols - 1, dtype=jnp.float32)
    single_view = jnp.ones((vol_size, vol_size), dtype=jnp.float32)

    @jax.jit
    def filter_one_view(view):
        return jax.vmap(lambda row: jnp.convolve(row, recon_filter, mode='same'))(view)

    try:
        lowered = filter_one_view.lower(single_view)
        hlo_text = lowered.as_text()
        out_path = hlo_dir / f'filter_kernel_vol{vol_size}.txt'
        out_path.write_text(hlo_text)
        dumped += 1
    except Exception:
        pass

    return dumped


def main_file_trace(vol_sizes, trace_base, hlo_dir, results):
    """File-based tracing mode: captures traces to disk via jax.profiler.trace."""
    for vol_size in vol_sizes:
        print(f"\n{'=' * 60}")
        print(f"Volume size: {vol_size}\u00b3 (views: {vol_size // 2})")
        print("=" * 60)

        trace_dir = trace_base / f'vol{vol_size}'
        trace_dir.mkdir(parents=True, exist_ok=True)
        all_runs_ctx = jax.profiler.trace(str(trace_dir)) if TRACE_ALL else contextlib.nullcontext()

        with all_runs_ctx:
            for run in range(RUNS_PER_SIZE):
                if not TRACE_ALL and run == TRACE_RUN:
                    per_run_dir = trace_base / f'vol{vol_size}'
                    per_run_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n  Run {run + 1}/{RUNS_PER_SIZE} [TRACED -> {per_run_dir.relative_to(OUTPUT_DIR)}]")
                    per_run_ctx = jax.profiler.trace(str(per_run_dir))
                else:
                    per_run_ctx = contextlib.nullcontext()
                    label = "warmup+compile" if run == 0 else ""
                    suffix = f" [{label}]" if label else ""
                    traced_label = " [TRACED]" if TRACE_ALL else ""
                    print(f"\n  Run {run + 1}/{RUNS_PER_SIZE}{suffix}{traced_label}")

                with per_run_ctx:
                    timings = run_profiling_pass(vol_size)

                for operation, elapsed in timings.items():
                    results['measurements'].append({
                        'operation': operation,
                        'volume_size': vol_size,
                        'run': run + 1,
                        'time': elapsed,
                        'traced': TRACE_ALL or run == TRACE_RUN,
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


def main_server_mode(vol_sizes, hlo_dir, results, server_port=9012):
    """Server-based profiling mode: starts a profiler gRPC server so XProf
    can do a live CAPTURE PROFILE.  This gives XProf full control over CUPTI
    data collection and produces the richest profiling data (HLO Op Profile,
    Roofline, etc.).

    Workflow:
      1. Script starts profiler server and does JIT warmup.
      2. TensorBoard is launched separately (or is already running).
      3. User clicks CAPTURE PROFILE in XProf, entering the server address.
      4. Script runs the profiling pass while XProf captures GPU data.
    """
    # Start profiler server BEFORE any JAX computation
    jax.profiler.start_server(server_port)
    print(f"\n  Profiler server started on port {server_port}")

    for vol_size in vol_sizes:
        print(f"\n{'=' * 60}")
        print(f"Volume size: {vol_size}\u00b3 (views: {vol_size // 2})")
        print("=" * 60)

        # Run 1: JIT warmup (compile all kernels)
        print(f"\n  Run 1/{RUNS_PER_SIZE} [warmup+compile]")
        timings = run_profiling_pass(vol_size)
        for op, elapsed in timings.items():
            results['measurements'].append({
                'operation': op, 'volume_size': vol_size,
                'run': 1, 'time': elapsed, 'traced': False,
            })
            print(f"    {op}: {elapsed:.4f}s")

        # Prompt user to start capture before the execution runs
        print(f"\n  ┌─────────────────────────────────────────────────┐")
        print(f"  │  Ready for XProf capture (vol {vol_size}\u00b3)              │")
        print(f"  │                                                 │")
        print(f"  │  In TensorBoard (http://localhost:6006):        │")
        print(f"  │    1. Go to the Profile tab                     │")
        print(f"  │    2. Click CAPTURE PROFILE                     │")
        print(f"  │    3. Profile Service URL: localhost:{server_port}       │")
        print(f"  │    4. Duration: 30000 ms                        │")
        print(f"  │    5. Click CAPTURE                             │")
        print(f"  │                                                 │")
        print(f"  │  Then press Enter here to start execution...    │")
        print(f"  └─────────────────────────────────────────────────┘")

        try:
            input()
        except EOFError:
            # Non-interactive: wait 5 seconds for capture to start
            print("  (non-interactive mode, starting in 5 seconds)")
            time.sleep(5)

        # Runs 2+: execution runs (XProf captures these)
        for run in range(1, RUNS_PER_SIZE):
            print(f"\n  Run {run + 1}/{RUNS_PER_SIZE} [CAPTURE ACTIVE]")
            timings = run_profiling_pass(vol_size)
            for op, elapsed in timings.items():
                results['measurements'].append({
                    'operation': op, 'volume_size': vol_size,
                    'run': run + 1, 'time': elapsed, 'traced': True,
                })
                print(f"    {op}: {elapsed:.4f}s")

        # Cost analysis + HLO dumps (outside capture window)
        print(f"\n  Cost analysis (vol {vol_size}\u00b3):")
        costs = collect_cost_analysis(vol_size)
        results['cost_analysis'][str(vol_size)] = costs

        print(f"\n  HLO dumps (vol {vol_size}\u00b3):")
        n_dumped = dump_hlo(vol_size, hlo_dir)
        print(f"    {n_dumped} HLO graphs saved to {hlo_dir.relative_to(OUTPUT_DIR)}/")


def main():
    parser = argparse.ArgumentParser(description='MBIRJAX GPU Profiler')
    parser.add_argument('--server', action='store_true',
                        help='Use profiler server mode (recommended). '
                             'Start TensorBoard separately and use CAPTURE PROFILE.')
    parser.add_argument('--port', type=int, default=9012,
                        help='Profiler server port (default: 9012)')
    args = parser.parse_args()

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
    print(f"Runs/size:    {RUNS_PER_SIZE}")
    print(f"Mode:         {'server (live capture)' if args.server else 'file trace'}")

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

    if args.server:
        main_server_mode(VOLUME_SIZES, hlo_dir, results, args.port)
    else:
        main_file_trace(VOLUME_SIZES, trace_base, hlo_dir, results)

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
    if not args.server:
        print(f"  XLA traces:            jax_traces/{timestamp}/")
    print(f"  HLO dumps:             hlo_dumps/{timestamp}/")
    if not args.server:
        print(f"\nView traces in TensorBoard:")
        print(f"  tensorboard --logdir=/output/jax_traces/{timestamp}")
    print(f"\nXProf tools to use:")
    print(f"  - Trace Viewer:     GPU compute vs memory transfer timeline")
    print(f"  - Roofline Analysis: identify memory-bound ops (FPGA candidates)")
    print(f"  - HLO Op Stats:     per-operation time and arithmetic intensity")
    print(f"  - Memory Viewer:    buffer sizes for FPGA on-chip memory planning")


if __name__ == '__main__':
    main()
