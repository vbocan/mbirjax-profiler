#!/usr/bin/env python3
"""
Python Profiler for MBIRJAX demos.

Usage:
    python profiling_wrapper.py /demos/demo_1_shepp_logan.py

Output:
    - *_python.txt   : Python function timing (text)
    - *_python.prof  : Binary profile (open with snakeviz)
"""

import sys
import os
import cProfile
import pstats
from pathlib import Path
from unittest.mock import MagicMock

# ============================================================================
# CONFIGURATION
# ============================================================================

DEMO_PATH = sys.argv[1] if len(sys.argv) > 1 else "/demos/demo_6_qggmrf_denoiser.py"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output")) / "profiles"
TOP_N = 30  # Number of top functions to show

# ============================================================================
# SETUP - Block GUI dependencies
# ============================================================================

sys.modules["easygui"] = MagicMock()
import mbirjax as mj
mj.slice_viewer = lambda *args, **kwargs: None

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import runpy
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_name = Path(DEMO_PATH).stem
    
    print(f"Demo: {DEMO_PATH}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Run with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    runpy.run_path(DEMO_PATH, run_name="__main__")
    profiler.disable()
    
    # Save binary profile (for snakeviz)
    prof_path = OUTPUT_DIR / f"{demo_name}_python.prof"
    profiler.dump_stats(str(prof_path))
    
    # Save text report
    report_path = OUTPUT_DIR / f"{demo_name}_python.txt"
    with open(report_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(TOP_N)
    
    # Print to console
    print("\n" + "=" * 60)
    print("PROFILE RESULTS")
    print("=" * 60)
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(TOP_N)
    
    # Print output summary and visualization instructions
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  {report_path}")
    print(f"  {prof_path}")
    print(f"  /output/hlo_dumps_xla/*.html")
    
    print("\n" + "=" * 60)
    print("HOW TO VISUALIZE")
    print("=" * 60)
    print()
    print("Install snakeviz (if needed): pip install snakeviz")
    print()
    print("Python profile (CPU hotspots):")
    print(f"  cat ./output/profiles/{demo_name}_python.txt")
    print(f"  snakeviz ./output/profiles/{demo_name}_python.prof")
    print()
    print("HLO graphs (GPU/XLA operations):")
    print("  Open ./output/hlo_dumps_xla/*.html in browser")
