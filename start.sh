#!/usr/bin/env bash
# MBIRJAX Profiler — Linux/macOS launcher
# Provides menu-driven access to GPU/CPU profiling and TensorBoard visualization
set -euo pipefail

IMAGE_GPU="mbirjax-profiler:gpu"
IMAGE_CPU="mbirjax-profiler:cpu"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/output"

# --- Colors ---
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
DIM='\033[2m'
RESET='\033[0m'

print_header() {
    clear
    echo ""
    echo -e "  ${CYAN}MBIRJAX Profiler${RESET}"
    echo -e "  ${DIM}================${RESET}"
    echo ""
}

image_exists() {
    [[ -n "$(docker images -q "$1" 2>/dev/null)" ]]
}

build_image() {
    local mode="$1"
    echo ""
    if [[ "$mode" == "gpu" ]]; then
        echo -e "  ${YELLOW}Building GPU image...${RESET}"
        docker build -t "$IMAGE_GPU" "$SCRIPT_DIR"
    else
        echo -e "  ${YELLOW}Building CPU image...${RESET}"
        docker build \
            --build-arg BASE_IMAGE=ubuntu:22.04 \
            --build-arg JAX_PACKAGE=jax \
            --build-arg JAX_PLATFORMS_DEFAULT=cpu \
            --build-arg XLA_FLAGS_DEFAULT="--xla_dump_to=/output/hlo_dumps_xla --xla_dump_hlo_as_text --xla_dump_hlo_as_html=true" \
            -t "$IMAGE_CPU" "$SCRIPT_DIR"
    fi

    if [[ $? -ne 0 ]]; then
        echo -e "  ${RED}Build failed.${RESET}"
        return 1
    fi
    echo -e "  ${GREEN}Build complete.${RESET}"
}

ensure_image() {
    local mode="$1"
    local image
    if [[ "$mode" == "gpu" ]]; then image="$IMAGE_GPU"; else image="$IMAGE_CPU"; fi

    if image_exists "$image"; then
        return 0
    fi

    echo -e "  ${YELLOW}Image '$image' not found.${RESET}"
    read -rp "  Build it now? [Y/n] " reply
    if [[ "$reply" =~ ^[nN] ]]; then
        return 1
    fi
    build_image "$mode"
}

run_profiler() {
    local mode="$1"

    if ! ensure_image "$mode"; then return; fi

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo -e "  ${CYAN}Running profiler ($mode mode)...${RESET}"
    echo ""

    if [[ "$mode" == "gpu" ]]; then
        docker run --rm --gpus all --shm-size=16g \
            --cap-add=SYS_ADMIN \
            --security-opt=seccomp:unconfined \
            --ulimit memlock=-1:-1 \
            -v "$OUTPUT_DIR:/output" \
            "$IMAGE_GPU"
    else
        docker run --rm \
            -v "$OUTPUT_DIR:/output" \
            "$IMAGE_CPU"
    fi

    echo ""
    if [[ $? -eq 0 ]]; then
        echo -e "  ${GREEN}Profiling complete. Results in: $OUTPUT_DIR${RESET}"
    else
        echo -e "  ${RED}Profiler exited with errors.${RESET}"
    fi
}

start_tensorboard() {
    # Use whichever image is available (prefer GPU)
    local image=""
    if image_exists "$IMAGE_GPU"; then
        image="$IMAGE_GPU"
    elif image_exists "$IMAGE_CPU"; then
        image="$IMAGE_CPU"
    fi

    if [[ -z "$image" ]]; then
        echo -e "  ${RED}No profiler image found. Build one first (option B).${RESET}"
        return
    fi

    if [[ ! -d "$OUTPUT_DIR/jax_traces" ]]; then
        echo -e "  ${RED}No traces found in $OUTPUT_DIR/jax_traces${RESET}"
        echo -e "  ${YELLOW}Run the profiler first to generate traces.${RESET}"
        return
    fi

    echo ""
    echo -e "  ${CYAN}Starting TensorBoard on http://localhost:6006${RESET}"
    echo -e "  ${DIM}Press Ctrl+C to stop.${RESET}"
    echo ""

    docker run --rm -p 6006:6006 \
        -v "$OUTPUT_DIR:/output" \
        "$image" \
        tensorboard --logdir=/output/jax_traces --host=0.0.0.0
}

# --- Main loop ---
while true; do
    print_header

    gpu_status="not built"; image_exists "$IMAGE_GPU" && gpu_status="ready"
    cpu_status="not built"; image_exists "$IMAGE_CPU" && cpu_status="ready"

    echo -e "  ${DIM}Images:  GPU [$gpu_status]  CPU [$cpu_status]${RESET}"
    echo ""
    echo -e "  ${YELLOW}[B] Build images${RESET}"
    echo -e "  ${GREEN}[G] Profile  (GPU)${RESET}"
    echo -e "  ${GREEN}[C] Profile  (CPU)${RESET}"
    echo -e "  ${MAGENTA}[V] View results  (TensorBoard)${RESET}"
    echo -e "  ${DIM}[Q] Quit${RESET}"
    echo ""

    read -rp "  Choice: " choice

    case "${choice^^}" in
        G) run_profiler "gpu" ;;
        C) run_profiler "cpu" ;;
        V) start_tensorboard ;;
        B)
            echo ""
            echo -e "  ${YELLOW}[1] GPU  [2] CPU  [3] Both${RESET}"
            read -rp "  Which: " sub
            case "$sub" in
                1) build_image "gpu" ;;
                2) build_image "cpu" ;;
                3) build_image "gpu"; build_image "cpu" ;;
                *) echo -e "  ${RED}Invalid choice.${RESET}" ;;
            esac
            ;;
        Q) echo ""; break ;;
        *) echo -e "  ${RED}Invalid choice.${RESET}" ;;
    esac

    echo ""
    read -rp "  Press Enter to continue..."
done
