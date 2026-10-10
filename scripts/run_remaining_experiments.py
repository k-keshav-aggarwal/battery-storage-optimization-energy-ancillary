#!/usr/bin/env python3
"""
scripts/run_remaining_experiments.py
====================================
Runs sinusoidal attack, breakeven analysis, and transfer experiments
to complete all artifacts in artifacts_v2.
"""

import subprocess
import sys
import time

def run_cmd(cmd):
    print("=" * 80)
    print(f"RUNNING: {' '.join(cmd)}")
    print("=" * 80)
    t0 = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"FAILED with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"COMPLETED in {time.time() - t0:.1f}s\n")

def main():
    run_cmd([sys.executable, "scripts/run_experiment.py", "--attack", "sinusoidal", "--epochs", "20"])
    run_cmd([sys.executable, "scripts/run_breakeven.py", "--epochs", "20"])
    run_cmd([sys.executable, "scripts/run_transfer.py", "--epochs", "20"])
    run_cmd([sys.executable, "scripts/generate_paper_plots.py"])
    print("ALL REMAINING EXPERIMENTS AND PLOTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
