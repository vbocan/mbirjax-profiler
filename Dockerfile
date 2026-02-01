# MBIRJAX Profiler with cProfile for FPGA Optimization Analysis
# CPU-only profiling for FPGA optimization research

FROM python:3.11-slim

LABEL maintainer="MBIRJAX Profiler Setup"
LABEL description="JAX-based tomographic reconstruction with Scalene CPU/memory profiling for FPGA optimization"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/output

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MBIRJAX (CPU version)
RUN pip install --no-cache-dir mbirjax

# Install snakeviz for visualizing cProfile results
RUN pip install --no-cache-dir snakeviz

# Create output directory
RUN mkdir -p ${OUTPUT_DIR}

# Copy profiling scripts
COPY scripts/ /scripts/
RUN chmod +x /scripts/*.py 2>/dev/null || true

# Set working directory
WORKDIR /output

# Default command
CMD ["/bin/bash"]
