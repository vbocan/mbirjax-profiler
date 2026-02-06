#!/bin/bash
# MBIRJAX Profiler - Raw timing data collection for FPGA analysis

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
    echo "  MBIRJAX Profiler"
    echo "============================================================"
    echo ""
    echo "  [R] Run profiler"
    echo "  [V] View profile with snakeviz"
    echo "  [Q] Quit"
    echo ""
}

run_profile() {
    echo ""
    echo "Running profiler..."
    echo "Volume sizes: 32, 64, 128, 256"
    echo "Runs per size: 3"
    echo ""

    docker-compose run --rm mbirjax-profiler python /scripts/comprehensive_profiler.py

    echo ""
    echo "[OK] Profiling completed"
    echo ""
    echo "Output:"
    echo "  JSON: output/mbirjax_profile_*.json"
    echo "  Prof: output/mbirjax_profile_*.prof"
    echo ""
    read -p "Press Enter to continue"
}

view_profile() {
    profiles=(output/*.prof)

    if [ ! -e "${profiles[0]}" ]; then
        echo ""
        echo "No profile files found. Run profiler first."
        echo ""
        read -p "Press Enter to continue"
        return
    fi

    echo ""
    echo "Available profiles:"
    echo ""

    i=1
    for prof in "${profiles[@]}"; do
        name=$(basename "$prof")
        size=$(du -k "$prof" | cut -f1)
        date=$(stat -c %y "$prof" 2>/dev/null || stat -f %Sm "$prof" 2>/dev/null)
        echo "  [$i] $name"
        echo "      ${size} KB  |  $date"
        ((i++))
    done

    echo ""
    read -p "Select profile number (or 'q' to cancel): " selection

    if [ "$selection" = "q" ]; then
        return
    fi

    if ! [[ "$selection" =~ ^[0-9]+$ ]] || [ "$selection" -lt 1 ] || [ "$selection" -gt "${#profiles[@]}" ]; then
        echo ""
        echo "Invalid selection"
        read -p "Press Enter to continue"
        return
    fi

    profile_file="${profiles[$((selection-1))]}"
    profile_name=$(basename "$profile_file")
    docker_path="/output/$profile_name"
    url="http://localhost:8080/snakeviz/$docker_path"

    echo ""
    echo "Starting snakeviz..."

    # Kill existing container on port 8080
    docker kill $(docker ps -q --filter "publish=8080") 2>/dev/null || true
    sleep 0.5

    docker run --rm -d -p 8080:8080 \
        -v "$(pwd)/output:/output" \
        mbirjax-profiler:latest \
        snakeviz -s -H 0.0.0.0 -p 8080 "$docker_path"

    sleep 3

    echo ""
    echo "snakeviz ready: $url"
    echo ""

    # Try to open browser
    xdg-open "$url" 2>/dev/null || open "$url" 2>/dev/null || echo "Open in browser: $url"

    echo "Stop with: docker kill \$(docker ps -q --filter 'publish=8080')"
    echo ""
    read -p "Press Enter to continue"
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice: " choice

    case "${choice^^}" in
        R) run_profile ;;
        V) view_profile ;;
        Q) echo ""; exit 0 ;;
        *) echo "Invalid choice"; read -p "Press Enter to continue" ;;
    esac
done
