# Scalene Profiler Assessment for FPGA Candidate Discovery

## Conclusion: Scalene is not effective for this use case

Scalene is a Python source-line profiler. It works by sampling the Python interpreter's execution state at intervals, attributing CPU time, memory allocations, and GPU utilization to Python source lines.

JAX does not execute Python code on the GPU. When a function like `model.forward_project(phantom)` is called, JAX traces the Python once, compiles it through XLA into GPU kernels (PTX/SASS), and on subsequent calls dispatches the cached compiled kernel directly — the Python code is never re-executed. Scalene sees a single Python line that blocks until the GPU finishes. It cannot see inside GPU kernels.

The evidence from our profiling runs confirms this. Scalene reported 0% Python / 0% native for nearly every MBIRJAX function. This is the correct result — JAX offloaded everything to GPU, and Scalene has no visibility into GPU execution.

Scalene's `--gpu` flag (which crashes with JAX CUDA due to signal-based sampling interfering with kernel execution) would only add coarse NVML-polled GPU utilization percentages per Python line. This would confirm that `forward_project` uses the GPU, but cannot reveal which XLA operations inside it are expensive — which is exactly what FPGA targeting requires.

### What works

The timing JSON from `time_operation()` + `block_until_ready()` provides accurate wall-clock time per MBIRJAX operation including GPU execution. This answers "which operations are slowest" — a necessary first step. The 28 operations currently profiled across ParallelBeamModel, ConeBeamModel, QGGMRFDenoiser, and utilities give good coverage of the MBIRJAX API.

### What doesn't work

Scalene adds near-zero value. It is a Python profiler being asked to see inside GPU kernels, which it structurally cannot do. The `--cpu-only` fallback required to avoid CUDA crashes makes it profile only Python dispatch overhead, which is irrelevant for FPGA candidate identification.

## Next step: JAX's built-in profiler

To identify kernels suitable for FPGA offload, the profiler needs to see the XLA computation graph — the actual operations (matmul, conv, scatter, gather, reduce, FFT) with their shapes, data flow, and execution times. JAX provides this natively.

### jax.profiler.trace

JAX includes a built-in profiler that outputs TensorBoard-compatible traces showing XLA operations with kernel-level timing.

Usage:

```python
import jax

with jax.profiler.trace("/output/jax_trace"):
    sinogram = model.forward_project(phantom)
    sinogram.block_until_ready()
```

The trace directory can be viewed with TensorBoard:

```bash
pip install tensorboard-plugin-profile
tensorboard --logdir /output/jax_trace
```

This shows:
- XLA op-level timeline (which operations run, how long, in what order)
- GPU kernel durations and overlap
- Memory transfer events (host-to-device, device-to-host)
- Op-level memory allocation

### XLA HLO dump

The actual computation graph compiled by XLA can be dumped to disk for static analysis:

```bash
XLA_FLAGS="--xla_dump_to=/output/xla_dump --xla_dump_hlo_as_text" python script.py
```

This produces `.hlo` text files showing every operation, its input/output shapes, and the graph structure. This is the most direct view of what would need to be implemented on an FPGA.

### jax.make_jaxpr

For quick inspection without execution, JAX can print its intermediate representation:

```python
from jax import make_jaxpr
jaxpr = make_jaxpr(model.forward_project)(phantom)
print(jaxpr)
```

This shows the JAX primitive operations (dot_general, conv, scatter, etc.) that map to XLA HLO operations, along with their tensor shapes.

### Recommended approach

1. Use `jax.profiler.trace` to wrap each MBIRJAX operation and collect XLA-level traces
2. View traces in TensorBoard to identify the most time-consuming XLA operations
3. Dump XLA HLO for the top candidates to understand the exact computation graph
4. Use HLO analysis to evaluate FPGA suitability (regular data access patterns, parallelizable reductions, fixed-size tensor operations)

The existing timing JSON infrastructure (28 operations, 4 volume sizes, 3 runs each) provides the operation-level ranking. The JAX profiler adds the missing kernel-level detail needed to make FPGA implementation decisions.
