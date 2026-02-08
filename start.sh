#!/bin/bash
# MBIRJAX Profiler (Scalene GPU) - Line-level CPU/GPU/memory profiling

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
    echo "  MBIRJAX Profiler (Scalene GPU)"
    echo "============================================================"
    echo ""
    echo "  [R] Run profiler"
    echo "  [V] View Scalene profile"
    echo "  [Q] Quit"
    echo ""
}

run_profile() {
    echo ""
    echo "Running Scalene profiler with GPU..."
    echo "Volume sizes: 32, 64, 128, 256"
    echo "Runs per size: 3"
    echo ""

    timestamp=$(date +%Y%m%d_%H%M%S)
    json_file="scalene_profile_${timestamp}.json"
    html_file="scalene_profile_${timestamp}.html"

    # Run Scalene profiler (produces JSON)
    docker compose run --rm mbirjax-profiler python -m scalene run --gpu -o "/output/${json_file}" /scripts/comprehensive_profiler.py

    if [ $? -ne 0 ]; then
        echo ""
        echo "[FAIL] Profiling failed"
        read -p "Press Enter to continue"
        return
    fi

    # Convert JSON profile to self-contained HTML
    echo ""
    echo "Generating HTML report..."
    docker compose run --rm mbirjax-profiler python -m scalene view --standalone "/output/${json_file}"

    # Rename the generated HTML to include timestamp
    if [ -f "output/scalene-profile.html" ]; then
        mv "output/scalene-profile.html" "output/${html_file}"
    fi

    echo ""
    echo "[OK] Profiling completed"
    echo ""
    echo "Output:"
    echo "  Scalene HTML: output/${html_file}"
    echo "  Scalene JSON: output/${json_file}"
    echo "  Timing JSON:  output/mbirjax_profile_*.json"
    echo ""
    read -p "Press Enter to continue"
}

view_profile() {
    profiles=(output/scalene_profile_*.html)

    if [ ! -e "${profiles[0]}" ]; then
        echo ""
        echo "No Scalene HTML profiles found. Run profiler first."
        echo ""
        read -p "Press Enter to continue"
        return
    fi

    echo ""
    echo "Available Scalene profiles:"
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
    full_path="$(pwd)/$profile_file"

    echo ""
    echo "Opening in browser: $(basename "$profile_file")"

    xdg-open "$full_path" 2>/dev/null || open "$full_path" 2>/dev/null || echo "Open in browser: $full_path"

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
