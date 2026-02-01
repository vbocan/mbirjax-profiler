#!/usr/bin/env python3
"""
Analyze cProfile output and recommend FPGA optimization candidates.

Helps identify functions suitable for FPGA acceleration based on:
- Cumulative execution time
- Call frequency
- Function characteristics
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict


def parse_cprofile_txt(filepath: str) -> List[Dict]:
    """Parse cProfile .txt report and extract function statistics."""
    functions = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the start of the function list
    in_table = False
    for line in lines:
        if 'function calls' in line:
            in_table = True
            continue

        if not in_table:
            continue

        # Skip empty lines and the header separator
        if not line.strip() or line.startswith('---'):
            continue

        # Parse data lines (format: ncalls tottime percall cumtime percall filename:lineno(function))
        # Handle both normal and primitive call formats
        match = re.match(
            r'\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(.+)',
            line
        )

        if match:
            ncalls, tottime, percall_tot, cumtime, percall_cum, location = match.groups()

            # Extract function name and file
            # Format: /path/file.py:line(function_name) or built-in functions
            if '(' in location:
                func_match = re.match(r'(.+):(\d+)\(([^)]+)\)', location)
                if func_match:
                    filepath_str, lineno, funcname = func_match.groups()
                    functions.append({
                        'function': funcname,
                        'file': filepath_str,
                        'lineno': lineno,
                        'ncalls': int(ncalls),
                        'tottime': float(tottime),
                        'cumtime': float(cumtime),
                        'percall_cum': float(percall_cum),
                        'location': location,
                    })
            else:
                # Built-in functions or C extensions
                functions.append({
                    'function': location.strip(),
                    'file': '(built-in)',
                    'lineno': '0',
                    'ncalls': int(ncalls),
                    'tottime': float(tottime),
                    'cumtime': float(cumtime),
                    'percall_cum': float(percall_cum),
                    'location': location,
                })

    return functions


def score_fpga_candidate(func: Dict) -> float:
    """
    Score a function as FPGA candidate (0-100).

    Considers:
    - Cumulative time (main factor)
    - Call frequency (repetitive work = good candidate)
    - Per-call time (expensive operations = worth accelerating)
    """
    score = 0.0

    # Cumulative time is primary factor (max 50 points)
    # Higher cumtime = higher priority
    score += min(func['cumtime'] * 10, 50)

    # Call frequency bonus (max 30 points)
    # Many calls = repetitive work = good FPGA candidate
    if func['ncalls'] > 100:
        score += min((func['ncalls'] / 1000) * 30, 30)

    # Per-call time bonus (max 20 points)
    # Expensive per-call = worth accelerating
    if func['percall_cum'] > 0.01:
        score += min(func['percall_cum'] * 1000, 20)

    return score


def categorize_function(func: Dict, all_functions: List[Dict]) -> str:
    """Categorize function as potential optimization candidate."""
    # JAX/NumPy operations - often good candidates
    if any(x in func['function'] for x in ['dot', 'matmul', 'multiply', 'add', 'conv', 'transpose']):
        return "MATH_OPS"

    # Scientific computing - often good candidates
    if any(x in func['function'] for x in ['radon', 'reconstruction', 'project', 'backproject', 'fft']):
        return "COMPUTE_INTENSIVE"

    # Memory operations - sometimes candidates
    if any(x in func['function'] for x in ['copy', 'reshape', 'slice', 'gather', 'scatter']):
        return "MEMORY_BOUND"

    # Control flow - usually not good candidates
    if any(x in func['function'] for x in ['if', 'while', 'for', 'range', 'enumerate']):
        return "CONTROL_FLOW"

    # I/O operations - not candidates
    if any(x in func['function'] for x in ['read', 'write', 'open', 'close', 'print', 'format']):
        return "IO_BOUND"

    # Python interpreter - not candidates
    if func['file'] == '(built-in)' or 'python' in func['file'].lower():
        return "INTERPRETER"

    # Default: unknown
    if func['cumtime'] > 0.1 and func['ncalls'] > 10:
        return "HOTSPOT"

    return "LOW_PRIORITY"


def analyze_profile(filepath: str, top_n: int = 20):
    """Analyze cProfile output and print FPGA recommendations."""

    print("\n" + "=" * 90)
    print("MBIRJAX cProfile Analysis - FPGA Optimization Candidates")
    print("=" * 90)

    # Parse profile
    functions = parse_cprofile_txt(filepath)

    if not functions:
        print("ERROR: Could not parse profile data from", filepath)
        return

    # Score and sort
    for func in functions:
        func['fpga_score'] = score_fpga_candidate(func)

    functions.sort(key=lambda x: x['cumtime'], reverse=True)

    # Print summary
    total_time = sum(f['cumtime'] for f in functions)
    total_calls = sum(f['ncalls'] for f in functions)

    print(f"\nProfile Summary:")
    print(f"  Total functions: {len(functions)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total calls: {total_calls:,}")

    # Print category legend
    print("\n" + "=" * 110)
    print("CATEGORY LEGEND:")
    print("=" * 110)
    print("  MATH_OPS           - Linear algebra operations (good for optimization)")
    print("  COMPUTE_INTENSIVE  - Heavy computation (excellent targets)")
    print("  MEMORY_BOUND       - Memory operations (sometimes worth optimizing)")
    print("  HOTSPOT            - Expensive functions worth investigating")
    print("  CONTROL_FLOW       - Control flow (not good targets)")
    print("  IO_BOUND           - I/O operations (not worth optimizing)")
    print("  INTERPRETER        - Python interpreter overhead")
    print("  LOW_PRIORITY       - Minor contributor to runtime")

    # Print top candidates
    print(f"\n{'RANK':<6} {'CATEGORY':<20} {'FUNCTION':<35} {'CUMTIME':<12} {'CALLS':<10} {'SCORE':<8}")
    print("-" * 110)

    candidates = []
    for i, func in enumerate(functions[:top_n], 1):
        category = categorize_function(func, functions)

        # Skip low-priority items
        if func['cumtime'] < 0.001:
            continue

        time_pct = (func['cumtime'] / total_time * 100) if total_time > 0 else 0

        func_display = func['function'][:34]
        score_display = f"{func['fpga_score']:.0f}/100"

        print(f"{i:<6} {category:<20} {func_display:<35} {func['cumtime']:>10.4f}s  {func['ncalls']:>8,d}  {score_display:<8}")

        # Track potential candidates
        if func['cumtime'] > total_time * 0.01 and func['ncalls'] > 10:  # >1% of time or frequently called
            candidates.append((i, func, category))

    # Recommendations
    print("\n" + "=" * 90)
    print("FPGA OPTIMIZATION RECOMMENDATIONS")
    print("=" * 90)

    if candidates:
        print("\nHigh-Priority Targets (top 5 by cumulative time):")
        for rank, func, category in candidates[:5]:
            pct = (func['cumtime'] / total_time * 100) if total_time > 0 else 0
            print(f"\n{rank}. {func['function']} ({category})")
            print(f"   Location: {func['file']}:{func['lineno']}")
            print(f"   Cumulative Time: {func['cumtime']:.4f}s ({pct:.1f}% of total)")
            print(f"   Calls: {func['ncalls']:,} (avg {func['percall_cum']:.6f}s/call)")
            print(f"   Recommendation:", end=" ")

            # Generate specific recommendation
            if "project" in func['function'].lower() or "radon" in func['function'].lower():
                print("Forward projection is a classic FPGA target. Consider accelerating radon transform.")
            elif "reconstruction" in func['function'].lower() or "recon" in func['function'].lower():
                print("MBIR reconstruction algorithm - major bottleneck. High ROI for FPGA acceleration.")
            elif any(x in func['function'] for x in ['dot', 'matmul', 'multiply']):
                print("Linear algebra operation - excellent FPGA candidate. High parallelism available.")
            elif func['ncalls'] > 10000:
                print("Extremely high call frequency - excellent candidate for pipelining in FPGA.")
            elif func['cumtime'] > 0.5:
                print("Significant time consumer - worth investigating for acceleration potential.")
            else:
                print("Monitor this function - may become bottleneck after accelerating others.")
    else:
        print("\nNo clear candidates found. Try profiling with larger datasets to identify hotspots.")

    # Analysis tips
    print("\n" + "-" * 90)
    print("ANALYSIS TIPS:")
    print("-" * 90)
    print("""
1. FOCUS on functions with:
   - High cumulative time (>1% of total)
   - High call frequency (>1000 calls)
   - Regular/predictable operations

2. LOOK for patterns:
   - Forward/back projection operations (good FPGA candidates)
   - Matrix operations (excellent for FPGA)
   - Iterative loops (check the body, not the loop control)

3. CONSIDER data flow:
   - What data flows through these functions?
   - Can computation be pipelined?
   - What are the memory access patterns?

4. NEXT STEPS:
   - Profile with different problem sizes (64³ vs 256³ vs 512³)
   - Look for functions that scale non-linearly (indicate algorithmic issues)
   - Consider memory bandwidth: is data being transferred efficiently?
   - Profile on GPU to see what JAX accelerates naturally vs what stays on CPU
    """)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_profile.py <profile.txt> [top_n_functions]")
        print("Example: python analyze_profile.py mbirjax_v256_n180_i10_20250131_143022.txt 20")
        sys.exit(1)

    profile_file = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    if not Path(profile_file).exists():
        print(f"ERROR: Profile file not found: {profile_file}")
        sys.exit(1)

    analyze_profile(profile_file, top_n)


if __name__ == '__main__':
    main()
