# MBIRJAX Profiler — cProfile + HLO dumps for FPGA candidate discovery
# Supports GPU (CUDA) and CPU-only modes via build args
#
# Usage (via docker compose):
#   Build: docker compose build gpu
#   Run:   docker compose run --rm gpu python /scripts/profiling_wrapper.py /demos/demo_1_shepp_logan.py
#
# Usage (standalone):
#   GPU:  docker run --rm --gpus all -v ./output:/output mbirjax-profiler:gpu
#   CPU:  docker run --rm -v ./output:/output mbirjax-profiler:cpu

ARG BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

LABEL maintainer="MBIRJAX Profiler Setup"
LABEL description="JAX-based tomographic reconstruction profiling for FPGA optimization"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/output
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install Python 3.11 via deadsnakes PPA (Ubuntu 22.04 ships 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gpg \
    gpg-agent \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip

# Mode-specific defaults (overridden by build args for CPU image)
ARG JAX_PLATFORMS_DEFAULT="cuda,cpu"
ARG XLA_FLAGS_DEFAULT="--xla_dump_to=/output/hlo_dumps_xla --xla_dump_hlo_as_text --xla_dump_hlo_as_html=true --xla_hlo_profile --xla_gpu_deterministic_ops"
ENV JAX_PLATFORMS=${JAX_PLATFORMS_DEFAULT}
ENV XLA_FLAGS=${XLA_FLAGS_DEFAULT}
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV JAX_TRACEBACK_FILTERING=off

# Install JAX (CUDA or CPU-only) and MBIRJAX
ARG JAX_PACKAGE="jax[cuda12]"
RUN pip install --no-cache-dir ${JAX_PACKAGE} mbirjax

# Create output directory
RUN mkdir -p ${OUTPUT_DIR}

# Copy demo scripts and profiling wrapper
COPY demos/ /demos/
COPY scripts/ /scripts/

WORKDIR /output

CMD ["python", "/scripts/profiling_wrapper.py", "/demos/demo_6_qggmrf_denoiser.py"]
