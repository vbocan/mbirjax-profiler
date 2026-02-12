# MBIRJAX Profiler — XLA-level profiling for FPGA candidate discovery
# Supports GPU (CUDA devel + CUPTI) and CPU-only modes via build args

ARG BASE_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

LABEL maintainer="MBIRJAX Profiler Setup"
LABEL description="JAX-based tomographic reconstruction with XLA-level GPU profiling for FPGA optimization"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/output

# Install Python 3.11 via deadsnakes PPA (Ubuntu 22.04 ships 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python/python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir --upgrade pip

# Install JAX (CUDA or CPU-only), MBIRJAX, and profiling tools
ARG JAX_PACKAGE="jax[cuda12]"
RUN pip install --no-cache-dir ${JAX_PACKAGE} mbirjax tensorboard xprof

# Create output directory
RUN mkdir -p ${OUTPUT_DIR}

# Copy profiling scripts
COPY scripts/ /scripts/
RUN chmod +x /scripts/*.py 2>/dev/null || true

# Set working directory
WORKDIR /output

# Default command
CMD ["/bin/bash"]
