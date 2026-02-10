#!/bin/bash
# MBIRJAX GPU Profiler — XLA-level profiling for FPGA candidate discovery

set -e
cd "$(dirname "$0")"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

show_menu() {
    clear
    echo ""
    echo "============================================================"
    echo "  MBIRJAX GPU Profiler (XLA / TensorBoard)"
    echo "============================================================"
    echo ""
    echo "  [R] Run profiler  (captures XLA traces + cost analysis + HLO)"
    echo "  [T] TensorBoard   (view XLA traces in browser)"
    echo "  [Q] Quit"
    echo ""
}

run_profile() {
    echo ""
    echo "Running GPU profiler..."
    echo "Volume sizes: 32, 64, 128, 256"
    echo "Runs per size: 3 (run 2 is traced)"
    echo ""

    docker compose run --rm mbirjax-profiler python /scripts/comprehensive_profiler.py

    if [ $? -ne 0 ]; then
        echo ""
        echo "[FAIL] Profiling failed"
        read -p "Press Enter to continue"
        return
    fi

    echo ""
    echo "[OK] Profiling completed"
    echo ""
    echo "Output:"
    echo "  Timing + cost: output/mbirjax_profile_*.json"
    echo "  XLA traces:    output/jax_traces/"
    echo "  HLO dumps:     output/hlo_dumps/"
    echo ""
    echo "Next: press [T] to launch TensorBoard"
    echo ""
    read -p "Press Enter to continue"
}

start_tensorboard() {
    # Find trace directories
    if [ ! -d "output/jax_traces" ]; then
        echo ""
        echo "No XLA traces found. Run the profiler first."
        echo ""
        read -p "Press Enter to continue"
        return
    fi

    sessions=(output/jax_traces/*/)
    if [ ! -d "${sessions[0]}" ]; then
        echo ""
        echo "No XLA traces found. Run the profiler first."
        echo ""
        read -p "Press Enter to continue"
        return
    fi

    echo ""
    echo "Available trace sessions:"
    echo ""

    # Sort descending (newest first)
    IFS=$'\n' sorted=($(printf '%s\n' "${sessions[@]}" | sort -r)); unset IFS

    i=1
    for session in "${sorted[@]}"; do
        name=$(basename "$session")
        vol_count=$(find "$session" -maxdepth 1 -type d | wc -l)
        vol_count=$((vol_count - 1))
        echo "  [$i] $name  ($vol_count volume sizes)"
        ((i++))
    done

    echo ""
    read -p "Select session (Enter for latest, 'q' to cancel): " selection

    if [ "$selection" = "q" ]; then
        return
    fi

    if [ -z "$selection" ]; then
        selected=$(basename "${sorted[0]}")
    elif [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#sorted[@]}" ]; then
        selected=$(basename "${sorted[$((selection-1))]}")
    else
        echo "Invalid selection"
        read -p "Press Enter to continue"
        return
    fi

    logdir="/output/jax_traces/$selected"

    echo ""
    echo "Launching TensorBoard..."
    echo "  Log dir: $logdir"
    echo "  URL:     http://localhost:6006"
    echo ""
    echo "Press Ctrl+C to stop TensorBoard"
    echo ""

    docker compose run --rm -p 6006:6006 mbirjax-profiler tensorboard --logdir="$logdir" --host=0.0.0.0 --port=6006
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice: " choice

    case "${choice^^}" in
        R) run_profile ;;
        T) start_tensorboard ;;
        Q) echo ""; exit 0 ;;
        *) echo "Invalid choice"; read -p "Press Enter to continue" ;;
    esac
done
