#!/bin/bash
# MBIRJAX Profiler — XLA-level profiling for FPGA candidate discovery

cd "$(dirname "$0")"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Default mode
MODE="gpu"
SERVICE="profiler-gpu"

show_menu() {
    clear
    echo ""
    echo "============================================================"
    echo "  MBIRJAX Profiler (XLA / TensorBoard)"
    echo "============================================================"
    echo ""
    if [ "$MODE" = "gpu" ]; then
        echo "  Mode: GPU  (CUDA + CUPTI profiling)"
    else
        echo "  Mode: CPU  (XLA traces + cost analysis only)"
    fi
    echo ""
    echo "  [R] Run profiler  (captures XLA traces + cost analysis + HLO)"
    echo "  [T] TensorBoard   (view XLA traces in browser)"
    echo "  [M] Switch mode   (CPU <-> GPU)"
    echo "  [Q] Quit"
    echo ""
}

switch_mode() {
    if [ "$MODE" = "gpu" ]; then
        MODE="cpu"
        SERVICE="profiler-cpu"
    else
        MODE="gpu"
        SERVICE="profiler-gpu"
    fi
    echo ""
    echo "Switched to ${MODE^^} mode"
    sleep 0.5
}

run_profile() {
    echo ""
    echo "Running profiler in ${MODE^^} mode..."
    echo ""

    if ! docker compose run --rm "$SERVICE" python /scripts/comprehensive_profiler.py; then
        echo ""
        echo "[FAIL] Profiling failed"
        read -p "Press Enter to continue"
        return
    fi

    echo ""
    echo "[OK] Profiling completed"
    echo ""
    echo "Output:"
    echo "  Timing + cost:  output/mbirjax_profile_*.json"
    echo "  XLA traces:     output/jax_traces/"
    echo "  HLO text:       output/hlo_dumps/"
    echo "  HLO graphs:     output/hlo_dumps_xla/*.html  (open in browser)"
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

    # List volume sizes in selected session
    vol_dirs=(output/jax_traces/$selected/*/)
    if [ ! -d "${vol_dirs[0]}" ]; then
        echo ""
        echo "No volume traces found in session $selected"
        read -p "Press Enter to continue"
        return
    fi

    IFS=$'\n' sorted_vols=($(printf '%s\n' "${vol_dirs[@]}" | sort)); unset IFS

    echo ""
    echo "Available volume sizes:"
    echo ""
    echo "  [A] All volumes (overview)"

    j=1
    for vol in "${sorted_vols[@]}"; do
        echo "  [$j] $(basename "$vol")"
        ((j++))
    done

    echo ""
    read -p "Select volume (Enter for all, 'q' to cancel): " vol_selection

    if [ "$vol_selection" = "q" ]; then
        return
    fi

    if [ -z "$vol_selection" ] || [ "${vol_selection^^}" = "A" ]; then
        logdir="/output/jax_traces/$selected"
    elif [[ "$vol_selection" =~ ^[0-9]+$ ]] && [ "$vol_selection" -ge 1 ] && [ "$vol_selection" -le "${#sorted_vols[@]}" ]; then
        vol_name=$(basename "${sorted_vols[$((vol_selection-1))]}")
        logdir="/output/jax_traces/$selected/$vol_name"
    else
        echo "Invalid selection"
        read -p "Press Enter to continue"
        return
    fi

    echo ""
    echo "Launching TensorBoard..."
    echo "  Log dir: $logdir"
    echo "  URL:     http://localhost:6006"
    echo ""
    echo "Press Ctrl+C to stop TensorBoard"
    echo ""

    # Open browser
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:6006" &
    elif command -v open &> /dev/null; then
        open "http://localhost:6006"
    fi

    docker compose run --rm -p 6006:6006 "$SERVICE" tensorboard --logdir="$logdir" --host=0.0.0.0 --port=6006
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice: " choice

    case "${choice^^}" in
        R) run_profile ;;
        T) start_tensorboard ;;
        M) switch_mode ;;
        Q) echo ""; exit 0 ;;
        *) echo "Invalid choice"; read -p "Press Enter to continue" ;;
    esac
done
