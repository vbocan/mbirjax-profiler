# MBIRJAX Profiler with Scalene for GPU/CPU/Memory Profiling
# CUDA runtime base — JAX bundles its own CUDA libs

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

LABEL maintainer="MBIRJAX Profiler Setup"
LABEL description="JAX-based tomographic reconstruction with Scalene GPU/CPU/memory profiling for FPGA optimization"

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

# Install JAX with CUDA support, MBIRJAX, and Scalene
RUN pip install --no-cache-dir "jax[cuda12]" mbirjax "scalene>=2.1.3"

# Create output directory
RUN mkdir -p ${OUTPUT_DIR}

# Copy profiling scripts
COPY scripts/ /scripts/
RUN chmod +x /scripts/*.py 2>/dev/null || true

# Set working directory
WORKDIR /output

# Default command
CMD ["/bin/bash"]
