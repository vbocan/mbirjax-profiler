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
    echo "  [R] Run profiler  (file-based trace)"
    echo "  [S] Server mode   (live XProf capture — best for HLO Op Profile)"
    echo "  [T] TensorBoard   (view XLA traces in browser)"
    echo "  [Q] Quit"
    echo ""
}

run_profile() {
    echo ""
    echo "Running GPU profiler (file-based trace)..."
    echo "Volume sizes: 32, 64, 128, 256"
    echo "Runs per size: 3 (all runs traced)"
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
    echo "  Timing + cost:  output/mbirjax_profile_*.json"
    echo "  XLA traces:     output/jax_traces/"
    echo "  HLO text:       output/hlo_dumps/"
    echo "  HLO graphs:     output/hlo_dumps_xla/*.html  (open in browser)"
    echo ""
    echo "Next: press [T] to launch TensorBoard"
    echo ""
    read -p "Press Enter to continue"
}

run_server_mode() {
    echo ""
    echo "Server mode: profiler + TensorBoard run together."
    echo "You'll use XProf's CAPTURE PROFILE for live GPU profiling."
    echo ""
    echo "Step 1: Starting TensorBoard + profiler server..."
    echo "Step 2: Open http://localhost:6006 in your browser"
    echo "Step 3: Follow the on-screen prompts"
    echo ""

    # Run profiler with --server flag, with both ports exposed.
    # Use -it for interactive input (Enter to proceed between volumes).
    docker compose run --rm \
        -p 6006:6006 \
        -p 9012:9012 \
        mbirjax-profiler \
        bash -c '
            # Start TensorBoard in the background
            tensorboard --logdir=/output/jax_traces --host=0.0.0.0 --port=6006 &
            TB_PID=$!
            sleep 3
            echo ""
            echo "TensorBoard running at http://localhost:6006"
            echo ""

            # Run profiler in server mode
            python /scripts/comprehensive_profiler.py --server --port 9012

            echo ""
            echo "Profiling complete. TensorBoard still running."
            echo "Check results in the Profile tab, then press Enter to exit."
            read
            kill $TB_PID 2>/dev/null
        '

    if [ $? -ne 0 ]; then
        echo ""
        echo "[FAIL] Server mode failed"
        read -p "Press Enter to continue"
        return
    fi

    echo ""
    echo "[OK] Server mode completed"
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

    docker compose run --rm -p 6006:6006 mbirjax-profiler tensorboard --logdir="$logdir" --host=0.0.0.0 --port=6006
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice: " choice

    case "${choice^^}" in
        R) run_profile ;;
        S) run_server_mode ;;
        T) start_tensorboard ;;
        Q) echo ""; exit 0 ;;
        *) echo "Invalid choice"; read -p "Press Enter to continue" ;;
    esac
done
