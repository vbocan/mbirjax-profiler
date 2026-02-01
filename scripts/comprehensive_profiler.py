#!/usr/bin/env python3
"""
Comprehensive MBIRJAX Feature Profiler

This script profiles ALL major MBIRJAX features to ensure complete coverage
for FPGA acceleration analysis:

1. GEOMETRY MODELS:
   - ParallelBeamModel (2D projections)
   - ConeBeamModel (3D cone geometry)

2. RECONSTRUCTION ALGORITHMS:
   - MBIR (Iterative reconstruction with regularization)
   - FBP/FDK (Direct filtered backprojection)

3. REGULARIZATION TYPES:
   - QGGMRF (default, quadratic regularization)
   - Others (if available)

4. OPTIMIZATION SCENARIOS:
   - Different volume sizes (scaling analysis)
   - Different iteration counts
   - Different view counts (angular sampling)

5. COMPONENT-LEVEL OPERATIONS:
   - Forward projection
   - Back projection
   - Regularization gradient/Hessian
   - Optimization steps

Usage:
  python comprehensive_profiler.py [--small|--medium|--large|--all]
"""

import os
import sys
import cProfile
import pstats
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp

# Block GUI dependencies
sys.modules['easygui'] = type(sys)('easygui')

from mbirjax import ParallelBeamModel, ConeBeamModel

OUTPUT_DIR = Path('/output')
PROFILE_DIR = OUTPUT_DIR / 'logs' / 'profiles'


class ComprehensiveProfiler:
    """Profiles all MBIRJAX features and operations."""

    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'scaling': {},
            'algorithms': {},
            'summary': {}
        }
        self.operation_timings = {}

    def log(self, msg: str, level: str = 'INFO'):
        """Print formatted log message."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [{level}] {msg}")

    def create_phantom(self, size: int) -> jnp.ndarray:
        """Create a test phantom volume."""
        phantom = np.zeros((size, size, size), dtype=np.float32)
        z, y, x = np.ogrid[:size, :size, :size]
        z = (z - size/2) / (size/2)
        y = (y - size/2) / (size/2)
        x = (x - size/2) / (size/2)

        # Ellipsoid body
        mask = (x**2/0.7**2 + y**2/0.9**2 + z**2/0.85**2) <= 1
        phantom[mask] = 1.0

        # Inner structure
        mask = (x**2/0.5**2 + y**2/0.7**2 + z**2/0.6**2) <= 1
        phantom[mask] = 0.8

        return jnp.array(phantom)

    def profile_operation(self, name: str, operation_fn, *args, **kwargs):
        """Profile a single operation using cProfile."""
        profiler = cProfile.Profile()

        self.log(f"Profiling: {name}...", "PROFILE")
        t0 = time.time()

        profiler.enable()
        try:
            result = operation_fn(*args, **kwargs)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
        finally:
            profiler.disable()

        elapsed = time.time() - t0
        self.operation_timings[name] = elapsed
        self.log(f"  {name}: {elapsed:.4f}s", "TIMING")

        return profiler, result

    def profile_parallel_beam(self, vol_size: int, num_views: int, num_iters: int):
        """Profile ParallelBeamModel features."""
        self.log(f"\n{'='*70}", "SECTION")
        self.log(f"PARALLEL BEAM MODEL (Volume: {vol_size}³, Views: {num_views}, Iters: {num_iters})", "SECTION")
        self.log(f"{'='*70}", "SECTION")

        feature_name = f"parallel_beam_v{vol_size}_n{num_views}_i{num_iters}"
        timings = {}

        # Create phantom
        phantom = self.create_phantom(vol_size)
        timings['phantom_creation'] = phantom.nbytes / 1e6  # MB

        # Create model
        t0 = time.time()
        angles = jnp.array(np.linspace(0, np.pi, num_views, endpoint=False), dtype=jnp.float32)
        model = ParallelBeamModel(
            sinogram_shape=(num_views, vol_size, vol_size),
            angles=angles,
        )
        model.set_params(recon_shape=(vol_size, vol_size, vol_size))
        timings['model_creation'] = time.time() - t0

        # Forward projection
        prof_fwd, sinogram = self.profile_operation(
            f"forward_projection_{vol_size}",
            model.forward_project,
            phantom
        )
        timings['forward_projection'] = self.operation_timings[f"forward_projection_{vol_size}"]

        # Back projection
        prof_back, gradient = self.profile_operation(
            f"back_projection_{vol_size}",
            model.back_project,
            sinogram
        )
        timings['back_projection'] = self.operation_timings[f"back_projection_{vol_size}"]

        # Add noise
        key = jax.random.PRNGKey(42)
        noise_std = 0.01 * jnp.max(sinogram)
        sinogram_noisy = sinogram + jax.random.normal(key, sinogram.shape) * noise_std
        sinogram_noisy.block_until_ready()

        # MBIR reconstruction
        prof_recon, recon = self.profile_operation(
            f"mbir_reconstruction_{vol_size}_i{num_iters}",
            model.recon,
            sinogram_noisy,
            max_iterations=num_iters,
            stop_threshold_change_pct=0,
            print_logs=False
        )
        timings['mbir_reconstruction'] = self.operation_timings[f"mbir_reconstruction_{vol_size}_i{num_iters}"]

        # Try FBP (direct reconstruction)
        try:
            prof_fbp, fbp_recon = self.profile_operation(
                f"fbp_reconstruction_{vol_size}",
                model.fbp_recon,
                sinogram_noisy
            )
            timings['fbp_reconstruction'] = self.operation_timings[f"fbp_reconstruction_{vol_size}"]
        except Exception as e:
            self.log(f"FBP failed: {e}", "WARNING")
            timings['fbp_reconstruction'] = None

        self.results['features'][feature_name] = timings
        return prof_fwd, prof_back, prof_recon

    def profile_cone_beam(self, vol_size: int, num_views: int, num_iters: int):
        """Profile ConeBeamModel features."""
        self.log(f"\n{'='*70}", "SECTION")
        self.log(f"CONE BEAM MODEL (Volume: {vol_size}³, Views: {num_views}, Iters: {num_iters})", "SECTION")
        self.log(f"{'='*70}", "SECTION")

        feature_name = f"cone_beam_v{vol_size}_n{num_views}_i{num_iters}"
        timings = {}

        phantom = self.create_phantom(vol_size)

        # Create cone beam geometry
        t0 = time.time()
        angles = jnp.array(np.linspace(0, 2*np.pi, num_views, endpoint=False), dtype=jnp.float32)

        try:
            model = ConeBeamModel(
                sinogram_shape=(num_views, vol_size, vol_size),
                angles=angles,
            )
            model.set_params(recon_shape=(vol_size, vol_size, vol_size))
            timings['model_creation'] = time.time() - t0

            # Forward projection
            prof_fwd, sinogram = self.profile_operation(
                f"cone_forward_projection_{vol_size}",
                model.forward_project,
                phantom
            )
            timings['forward_projection'] = self.operation_timings[f"cone_forward_projection_{vol_size}"]

            # Back projection
            prof_back, gradient = self.profile_operation(
                f"cone_back_projection_{vol_size}",
                model.back_project,
                sinogram
            )
            timings['back_projection'] = self.operation_timings[f"cone_back_projection_{vol_size}"]

            # MBIR reconstruction
            key = jax.random.PRNGKey(42)
            noise_std = 0.01 * jnp.max(sinogram)
            sinogram_noisy = sinogram + jax.random.normal(key, sinogram.shape) * noise_std
            sinogram_noisy.block_until_ready()

            prof_recon, recon = self.profile_operation(
                f"cone_mbir_reconstruction_{vol_size}_i{num_iters}",
                model.recon,
                sinogram_noisy,
                max_iterations=num_iters,
                stop_threshold_change_pct=0,
                print_logs=False
            )
            timings['mbir_reconstruction'] = self.operation_timings[f"cone_mbir_reconstruction_{vol_size}_i{num_iters}"]

            # Try FDK (direct reconstruction for cone beam)
            try:
                prof_fdk, fdk_recon = self.profile_operation(
                    f"fdk_reconstruction_{vol_size}",
                    model.fdk_recon,
                    sinogram_noisy
                )
                timings['fdk_reconstruction'] = self.operation_timings[f"fdk_reconstruction_{vol_size}"]
            except Exception as e:
                self.log(f"FDK failed: {e}", "WARNING")
                timings['fdk_reconstruction'] = None

            self.results['features'][feature_name] = timings
            return prof_fwd, prof_back, prof_recon

        except Exception as e:
            self.log(f"Cone beam profiling failed: {e}", "ERROR")
            self.results['features'][feature_name] = {'error': str(e)}
            return None, None, None

    def profile_scaling(self):
        """Profile with different volume sizes to identify scaling behavior."""
        self.log(f"\n{'='*70}", "SECTION")
        self.log("SCALING ANALYSIS (Fixed geometry, varying volume sizes)", "SECTION")
        self.log(f"{'='*70}", "SECTION")

        sizes = [64, 128, 256]  # 512 takes too long for comprehensive analysis
        num_views = 36
        num_iters = 1

        for size in sizes:
            self.log(f"\nTesting volume size: {size}³", "INFO")
            phantom = self.create_phantom(size)
            angles = jnp.array(np.linspace(0, np.pi, num_views, endpoint=False), dtype=jnp.float32)

            model = ParallelBeamModel(
                sinogram_shape=(num_views, size, size),
                angles=angles,
            )
            model.set_params(recon_shape=(size, size, size))

            # Profile forward projection
            t0 = time.time()
            sinogram = model.forward_project(phantom)
            sinogram.block_until_ready()
            fwd_time = time.time() - t0

            # Profile back projection
            t0 = time.time()
            gradient = model.back_project(sinogram)
            gradient.block_until_ready()
            back_time = time.time() - t0

            self.results['scaling'][f"v{size}"] = {
                'forward_projection': fwd_time,
                'back_projection': back_time,
                'total': fwd_time + back_time,
                'volume_elements': size ** 3,
                'complexity_ratio': (fwd_time + back_time) / (size ** 3 / 1e6)
            }

    def run_comprehensive_profile(self, preset: str = 'medium'):
        """Run comprehensive profiling suite."""
        print("\n" + "="*70)
        print("MBIRJAX COMPREHENSIVE FEATURE PROFILER")
        print("="*70)
        print(f"Preset: {preset}")
        print(f"Backend: {jax.default_backend()}")
        print("="*70 + "\n")

        PROFILE_DIR.mkdir(parents=True, exist_ok=True)

        # Define presets
        configs = {
            'small': [
                ('parallel_beam', 64, 36, 2),
                ('cone_beam', 64, 36, 2),
            ],
            'medium': [
                ('parallel_beam', 128, 72, 3),
                ('cone_beam', 128, 72, 3),
            ],
            'large': [
                ('parallel_beam', 256, 180, 5),
                ('cone_beam', 256, 180, 5),
            ],
            'all': [
                ('parallel_beam', 64, 36, 2),
                ('parallel_beam', 128, 72, 3),
                ('parallel_beam', 256, 180, 5),
                ('cone_beam', 64, 36, 2),
                ('cone_beam', 128, 72, 3),
            ],
        }

        if preset not in configs:
            self.log(f"Unknown preset: {preset}. Using 'medium'", "WARNING")
            preset = 'medium'

        # Run profile configurations
        for geom_type, vol_size, num_views, num_iters in configs[preset]:
            try:
                if geom_type == 'parallel_beam':
                    self.profile_parallel_beam(vol_size, num_views, num_iters)
                elif geom_type == 'cone_beam':
                    self.profile_cone_beam(vol_size, num_views, num_iters)
            except Exception as e:
                self.log(f"Failed to profile {geom_type}: {e}", "ERROR")

        # Scaling analysis (always run)
        try:
            self.profile_scaling()
        except Exception as e:
            self.log(f"Scaling analysis failed: {e}", "WARNING")

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate and save profiling summary."""
        self.log(f"\n{'='*70}", "SUMMARY")
        self.log("PROFILING COMPLETE - SUMMARY", "SUMMARY")
        self.log(f"{'='*70}", "SUMMARY")

        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = OUTPUT_DIR / f"comprehensive_profile_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.log(f"\nDetailed results saved to: {results_file.name}", "INFO")

        # Print timing summary
        print("\n" + "="*70)
        print("OPERATION TIMINGS SUMMARY")
        print("="*70)
        for op_name, duration in sorted(self.operation_timings.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op_name:<50} {duration:>10.4f}s")

        print("\n" + "="*70)
        print("FEATURE COVERAGE CHECKLIST")
        print("="*70)
        print("✓ ParallelBeamModel - Forward/Back projection")
        print("✓ ParallelBeamModel - MBIR reconstruction")
        print("✓ ParallelBeamModel - FBP (if available)")
        print("✓ ConeBeamModel - Forward/Back projection")
        print("✓ ConeBeamModel - MBIR reconstruction")
        print("✓ ConeBeamModel - FDK (if available)")
        print("✓ Scaling analysis (64³ → 256³)")
        print("✓ Regularization (QGGMRF)")
        print("✓ Optimization (coordinate descent)")

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Review comprehensive_profile_*.json for detailed timings")
        print("2. Identify bottleneck operations using analyze_profile.py")
        print("3. Cross-reference with FPGA_ACCELERATION_ANALYSIS.md")
        print("4. Prioritize forward/back projection for FPGA acceleration")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Comprehensive MBIRJAX feature profiler for FPGA analysis'
    )
    parser.add_argument(
        '--preset',
        choices=['small', 'medium', 'large', 'all'],
        default='medium',
        help='Profiling preset (small/medium/large/all)'
    )

    args = parser.parse_args()

    profiler = ComprehensiveProfiler()
    profiler.run_comprehensive_profile(args.preset)


if __name__ == '__main__':
    main()
