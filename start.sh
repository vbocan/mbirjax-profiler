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

build_images() {
    echo ""
    echo -e "  ${YELLOW}Building GPU and CPU images...${RESET}"
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" build
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
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" build "$mode"
}

run_profiler() {
    local mode="$1"

    if ! ensure_image "$mode"; then return; fi

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo -e "  ${CYAN}Running profiler ($mode mode)...${RESET}"
    echo ""

    docker compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm "$mode"

    echo ""
    if [[ $? -eq 0 ]]; then
        echo -e "  ${GREEN}Profiling complete. Results in: $OUTPUT_DIR${RESET}"
    else
        echo -e "  ${RED}Profiler exited with errors.${RESET}"
    fi
}

start_tensorboard() {
    if ! image_exists "$IMAGE_GPU" && ! image_exists "$IMAGE_CPU"; then
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

    docker compose -f "$SCRIPT_DIR/docker-compose.yml" up tensorboard
}

# --- Main loop ---
while true; do
    print_header

    gpu_status="not built"; image_exists "$IMAGE_GPU" && gpu_status="ready"
    cpu_status="not built"; image_exists "$IMAGE_CPU" && cpu_status="ready"

    echo -e "  ${DIM}Images:  GPU [$gpu_status]  CPU [$cpu_status]${RESET}"
    echo ""
    echo -e "  ${YELLOW}[B] Build images${RESET}"
    echo -e "  ${GREEN}[G] Profile using GPU${RESET}"
    echo -e "  ${GREEN}[C] Profile using CPU${RESET}"
    echo -e "  ${MAGENTA}[V] View results  (TensorBoard)${RESET}"
    echo -e "  ${DIM}[Q] Quit${RESET}"
    echo ""

    read -rp "  Choice: " choice

    case "${choice^^}" in
        G) run_profiler "gpu" ;;
        C) run_profiler "cpu" ;;
        V) start_tensorboard ;;
        B) build_images ;;
        Q) echo ""; break ;;
        *) echo -e "  ${RED}Invalid choice.${RESET}" ;;
    esac

    echo ""
    read -rp "  Press Enter to continue..."
done
