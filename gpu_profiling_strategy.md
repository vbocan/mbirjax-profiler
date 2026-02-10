# GPU Profiling Strategy for FPGA Candidate Discovery

## Background

The goal is to profile MBIRJAX GPU operations at the kernel and memory-transfer level to identify bottlenecks suitable for FPGA acceleration. The FPGA path requires:

1. Identifying memory transfers in GPU code (HtoD, DtoH, DtoD)
2. Defining a new XLA accelerator backend for the FPGA
3. Building a DMA driver to move data between host and FPGA
4. Implementing the compute kernels on the FPGA fabric

To design this, we need a clear picture of what the GPU actually does: compute vs memory transfer time, buffer sizes, data flow, and which operations are memory-bound (the best FPGA candidates).

## Why Scalene failed

JAX traces Python once, compiles through XLA into GPU kernels (PTX/SASS), and dispatches cached kernels on subsequent calls. Scalene samples the Python interpreter — it cannot see inside GPU kernels. It reported 0% Python / 0% native for nearly every MBIRJAX function. The `--gpu` flag crashes with JAX CUDA due to signal-based sampling conflicts, and would only provide coarse NVML-polled utilization percentages anyway. See `scalene_assessment.md` for full details.

## What already works

The timing JSON from `time_operation()` + `block_until_ready()` provides accurate wall-clock time per MBIRJAX operation (28 operations across ParallelBeamModel, ConeBeamModel, QGGMRFDenoiser, utilities; 4 volume sizes; 3 runs each). This answers "which operations are slowest" but not "why" or "what's happening inside the GPU."

---

## Profiling tools — by priority

### Tier 1: XLA-level profiling (start here)

#### 1. jax.profiler.trace + XProf/TensorBoard

The most important tool. Captures the actual XLA execution timeline on the GPU via CUPTI.

```python
import jax

with jax.profiler.trace("/output/jax_trace"):
    sinogram = model.forward_project(phantom)
    sinogram.block_until_ready()
```

```bash
pip install tensorboard xprof
tensorboard --logdir=/output/jax_trace
```

XProf views and their FPGA relevance:

| XProf Tool | What It Shows | FPGA Design Value |
|---|---|---|
| Trace Viewer | Timeline of compute streams vs memory copy streams, HtoD/DtoH events, idle gaps | Identifies memory transfer bottlenecks vs compute time |
| Roofline Analysis | Arithmetic intensity (FLOPS/byte) per operation, compute-bound vs memory-bound classification | Directly identifies memory-bound ops — the best FPGA candidates |
| HLO Op Stats | Per-operation time, GFLOPS/s, memory bandwidth, arithmetic intensity | Ranks operations by cost and compute/memory balance |
| GPU Kernel Stats | Per-kernel performance metrics mapped to JAX operations | Ground truth kernel timing |
| Memory Viewer | Buffer lifetimes and peak allocation contents | FPGA on-chip memory sizing |
| Memory Profile | Dynamic memory usage timeline | Allocation/deallocation patterns |
| Graph Viewer | Full HLO computation graph | Dataflow for FPGA pipeline design |

GPU-specific profiler options (via `jax.profiler.ProfileOptions`):
- `gpu_max_callback_api_events` — max CUPTI callback events (default 2M)
- `gpu_max_activity_api_events` — max CUPTI activity events (default 2M)
- `gpu_enable_nvtx_tracking` — enable NVTX tracking in CUPTI
- `gpu_enable_cupti_activity_graph_trace` — CUDA graph tracing
- `gpu_pm_sample_counters` — comma-separated GPU Performance Monitoring metrics
- `gpu_pm_sample_interval_us` — PM sampling interval (default 500us)

Note: By default, GPU traces prevent CUDA kernels from running concurrently, which gives more accurate per-kernel timings. For FPGA analysis, this is preferred.

#### 2. XLA HLO dump (static graph analysis)

Dumps the exact computation graph before and after XLA optimization/fusion. The after-optimization HLO is the closest representation of what the FPGA needs to implement.

```bash
# Text format (most readable)
XLA_FLAGS="--xla_dump_to=/output/hlo_dumps --xla_dump_hlo_as_text" python script.py

# HTML format (interactive browser viewing)
XLA_FLAGS="--xla_dump_to=/output/hlo_dumps --xla_dump_hlo_as_html=true" python script.py

# Protocol buffer format (machine-parseable)
XLA_FLAGS="--xla_dump_to=/output/hlo_dumps --xla_dump_hlo_as_proto" python script.py

# DOT graph format (visualize with graphviz)
XLA_FLAGS="--xla_dump_to=/output/hlo_dumps --xla_dump_hlo_as_dot" python script.py

# All formats at once
XLA_FLAGS="--xla_dump_to=/output/hlo_dumps --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_hlo_as_html=true" python script.py
```

What the dumps contain:
- Before-optimization HLO: the raw computation graph from JAX
- After-optimization HLO: the fused, optimized graph that actually runs on GPU
- Every intermediate compiler pass
- Operation types, tensor shapes, layouts, data dependencies

The after-optimization HLO shows XLA's fusion decisions. For FPGA design, the before-optimization graph shows logical operations, and the after-optimization graph shows what memory traffic the GPU actually experiences after fusion eliminates intermediate writes to HBM.

Additional XLA flags for analysis:
```bash
# Disable fusion to see individual operations (analysis only, not performance)
XLA_FLAGS="--xla_gpu_disable_multi_output_fuse=true"

# Enable latency-hiding scheduler analysis
XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"

# Dump all intermediate passes
XLA_FLAGS="--xla_dump_to=/tmp/xla --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"
```

#### 3. JAX AOT inspection (programmatic cost analysis)

JAX's three-stage compilation pipeline (trace -> lower -> compile) exposes inspection at each stage:

```python
import jax

# Stage 1: Jaxpr — JAX-level primitives with shapes
from jax import make_jaxpr
jaxpr = make_jaxpr(model.forward_project)(phantom)
print(jaxpr)

# Stage 2: Lowered — StableHLO/HLO text
lowered = jax.jit(model.forward_project).lower(phantom)
print(lowered.as_text())                # Human-readable StableHLO
print(lowered.as_text(debug_info=True)) # With source locations

# Stage 3: Compiled — cost and memory estimates
compiled = lowered.compile()
cost = compiled.cost_analysis()
if cost:
    print(f"FLOPs:          {cost[0].get('flops', 'N/A')}")
    print(f"Bytes accessed: {cost[0].get('bytes accessed', 'N/A')}")
    print(f"Bytes output:   {cost[0].get('bytes accessed output', 'N/A')}")
    print(f"Optimal secs:   {cost[0].get('optimal_seconds', 'N/A')}")

memory = compiled.memory_analysis()
if memory:
    print(f"Memory analysis: {memory}")
```

The ratio `flops / bytes_accessed` = arithmetic intensity. Low arithmetic intensity = memory-bound = strong FPGA candidate (FPGA custom memory hierarchies beat GPU HBM access patterns).

Caveat: FLOP counts are known to be incorrect for some operations (notably `dot_general`) on GPU. Use as order-of-magnitude estimates.

---

### Tier 2: NVIDIA hardware-level profiling

#### 4. NVIDIA Nsight Systems (nsys)

System-wide GPU profiler showing a timeline of all CUDA activity: kernel launches, memory copies (HtoD/DtoH/DtoD), CUDA API calls, PCIe throughput, DRAM activity.

Critical for JAX: XLA uses CUDA graphs by default, which hides individual kernel annotations. Disable them:
```bash
XLA_FLAGS="--xla_gpu_enable_command_buffer=" nsys profile --cuda-graph-trace=node python script.py
```

Targeted profiling (skip JIT warmup):
```python
from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')

# Warm up / JIT compile
for i in range(2):
    result = model.forward_project(phantom)
    result.block_until_ready()

# Profile only steady-state execution
libcudart.cudaProfilerStart()
result = model.forward_project(phantom)
result.block_until_ready()
libcudart.cudaProfilerStop()
```
```bash
nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node \
  --capture-range-end=stop python script.py
```

Custom annotations with NVTX:
```python
import nvtx

with nvtx.annotate("forward_project"):
    sinogram = model.forward_project(phantom)
    sinogram.block_until_ready()

# Inside JIT regions:
with jax.named_scope("my_operation"):
    result = some_jax_computation(x)
```

#### 5. nsys-jax (NVIDIA's JAX-specific wrapper)

Included in NVIDIA's JAX-Toolbox containers. Automatically collects JAX/XLA metadata and generates analysis-ready outputs.

```bash
nsys-jax python my_program.py
nsys-jax --nsys-jax-analysis python my_program.py  # with automatic analysis
nsys-jax-combine  # merge multi-process output
```

Produces: `.parquet` and `.csv.xz` files, Jupyter notebook with `nsys_jax` library for programmatic exploration.

#### 6. NVIDIA Nsight Compute (ncu)

Kernel-level profiler with detailed per-kernel metrics. Use only on the top 3-5 bottleneck kernels identified by Nsight Systems or XProf — it replays kernels multiple times and is very slow.

```bash
ncu --set full -k "kernel_name_regex" python script.py
```

Key metrics for FPGA design:
- Achieved memory bandwidth vs theoretical peak (memory-boundedness)
- Achieved FLOPS vs theoretical peak (compute-boundedness)
- Global memory load/store efficiency (coalescing quality)
- L1/L2 cache hit rates (locality — relevant for FPGA on-chip memory design)
- Occupancy (parallelism requirements)

---

### Tier 3: Memory architecture analysis

#### 7. Device memory profiling (pprof snapshots)

Captures a snapshot of all live device memory allocations attributed to the Python call stack.

```python
import jax

jax.profiler.save_device_memory_profile("/output/memory_before.prof")

result = model.forward_project(phantom)
result.block_until_ready()

jax.profiler.save_device_memory_profile("/output/memory_after.prof")
```

Viewing (requires Go and pprof):
```bash
go tool pprof -http=:8080 /output/memory_before.prof

# Diff between two snapshots
go tool pprof -base /output/memory_before.prof /output/memory_after.prof
```

Shows buffer sizes and allocation stacks — tells you how much on-chip FPGA memory you need.

#### 8. jax-smi (real-time memory monitoring)

Like nvidia-smi but shows actual JAX allocations within the pre-allocated memory pool (nvidia-smi always shows ~90% usage due to JAX pre-allocation).

```python
from jax_smi import initialise_tracking
initialise_tracking()
```
```bash
pip install jax-smi
jax-smi  # in another terminal
```

---

### Tier 4: FPGA handoff

#### 9. StableHLO export

JAX can export the computation graph in StableHLO format — a portable ML IR that can potentially be fed into FPGA HLS tools.

```python
import jax

exported = jax.export(jax.jit(model.forward_project))(phantom)
serialized = exported.mlir_module_serialized  # Portable IR for HLS tools
```

Research like LeFlow (https://arxiv.org/pdf/1807.05317) has demonstrated converting XLA LLVM output to HLS-compatible code for FPGA synthesis. The after-optimization HLO text dump is the most useful artifact for manual FPGA design.

---

## XLA internals relevant to FPGA design

**Fusion:** XLA's most important optimization. Groups multiple operations into a single kernel to avoid writing intermediate tensors to HBM. Many "operations" in the Jaxpr don't correspond to separate memory transfers on the GPU. The FPGA design can replicate or exceed this using on-chip memory.

**Emitters:** XLA:GPU has 7 emitter types, each corresponding to a "hero" operation in a fusion. The emitter type tells you the memory access pattern.

**Thunks and command buffers:** Runtime uses "thunks" (individual executable units) batched into CUDA graph command buffers. Disabling command buffers (`--xla_gpu_enable_command_buffer=`) gives individual thunk visibility in profilers.

**Memory coalescing:** XLA analyzes memory access coalescing patterns. Operations with poor coalescing benefit most from FPGA custom memory access patterns.

---

## Recommended phased workflow

| Phase | Tool | Question Answered |
|---|---|---|
| 1. Timeline | `jax.profiler.trace` + XProf | Which ops spend time on memory transfers vs compute? |
| 2. Graph | XLA HLO dump + `cost_analysis()` | What are the exact primitives, shapes, arithmetic intensity? |
| 3. Ground truth | Nsight Systems (`nsys`) | Is GPU memory bandwidth actually saturated? |
| 4. Deep dive | Nsight Compute (`ncu`) | Why are the top bottleneck kernels slow? |
| 5. Memory sizing | `save_device_memory_profile` + XProf Memory Viewer | How much FPGA on-chip memory is needed? |
| 6. FPGA handoff | StableHLO export + after-optimization HLO | What does the FPGA need to implement? |

---

## Docker integration

Add to `docker-compose.yml` environment:
```yaml
environment:
  # Existing
  - JAX_PLATFORMS=cuda,cpu
  - XLA_PYTHON_CLIENT_PREALLOCATE=true
  - XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
  # HLO dump for static analysis
  - XLA_FLAGS=--xla_dump_to=/output/hlo_dumps --xla_dump_hlo_as_text --xla_dump_hlo_as_html=true
  # Deeper Python tracebacks in profiler annotations
  - JAX_TRACEBACK_IN_LOCATIONS_LIMIT=-1
```

For Nsight Systems, add `cap_add: [SYS_ADMIN]` and install `nsys` in the container.

## Package requirements

```bash
pip install tensorboard xprof    # XProf / TensorBoard profiling
pip install jax-smi              # Real-time memory monitoring
pip install nvtx                 # NVTX annotations for nsys
go install github.com/google/pprof@latest  # pprof for memory profiles (requires Go)
```

## References

- JAX Profiling: https://docs.jax.dev/en/latest/profiling.html
- JAX Device Memory Profiling: https://docs.jax.dev/en/latest/device_memory_profiling.html
- JAX AOT Lowering: https://docs.jax.dev/en/latest/aot.html
- JAX GPU Performance Tips: https://docs.jax.dev/en/latest/gpu_performance_tips.html
- JAX XLA Compiler Flags: https://docs.jax.dev/en/latest/xla_flags.html
- XProf Profiler: https://openxla.org/xprof
- XProf HLO Op Stats: https://openxla.org/xprof/hlo_op_stats
- XLA GPU Architecture: https://openxla.org/xla/gpu_architecture
- XLA Emitters: https://openxla.org/xla/emitters
- XLA Tooling: https://openxla.org/xla/tools
- StableHLO Export: https://openxla.org/stablehlo/tutorials/jax-export
- NVIDIA JAX-Toolbox Profiling: https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/profiling.md
- nsys-jax: https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/nsys-jax.md
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems
- NVIDIA Nsight Compute: https://developer.nvidia.com/nsight-compute
- NVIDIA CUPTI: https://developer.nvidia.com/cupti
- All XLA Options: https://guides.lw1.at/all-xla-options/
- LeFlow (XLA to FPGA HLS): https://arxiv.org/pdf/1807.05317
