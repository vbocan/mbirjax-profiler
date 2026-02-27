#!/usr/bin/env bash
# MBIRJAX Profiler
#
# Usage:
#   ./profile.sh demo_1_shepp_logan.py
#
# Output: ./output/profiles/

set -euo pipefail
cd "$(dirname "$0")"
export MSYS_NO_PATHCONV=1

# Show help if no parameters
if [[ $# -eq 0 ]]; then
    echo "MBIRJAX Profiler"
    echo ""
    echo "Usage: ./profile.sh <demo_script>"
    echo ""
    echo "Available demos:"
    echo "  demo_1_shepp_logan.py              VCD Reconstruction (parallel beam)"
    echo "  demo_2_large_object.py             Large object (partial projection)"
    echo "  demo_3_cropped_center_recon.py     Cropped center reconstruction"
    echo "  demo_4_wrong_rotation_direction.py Wrong rotation direction (cone beam)"
    echo "  demo_5_fbp_fdk.py                  FBP Reconstruction"
    echo "  demo_6_qggmrf_denoiser.py          QGGMRF Denoising"
    echo ""
    echo "Example:"
    echo "  ./profile.sh demo_1_shepp_logan.py"
    echo ""
    echo "Output:"
    echo "  ./output/profiles/*_python.txt     cProfile text summary"
    echo "  ./output/profiles/*_python.prof    cProfile binary (snakeviz)"
    echo "  ./output/hlo_dumps_xla/*.html      XLA computation graphs"
    exit 0
fi

DEMO="$1"
DEMO_PATH="/demos/$DEMO"

# Build image if needed
IMAGE="mbirjax-profiler:gpu"
if [[ -z "$(docker images -q "$IMAGE" 2>/dev/null)" ]]; then
    echo "Building $IMAGE..."
    docker compose build gpu
fi

# Run profiler
echo "Profiling: $DEMO_PATH"
docker compose run --rm gpu python /scripts/profiling_wrapper.py "$DEMO_PATH"

echo -e "\nOutput: ./output/profiles/"
