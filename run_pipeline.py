#!/usr/bin/env python3
"""
run_pipeline.py — Victor One-Click Pipeline Runner

Runs the full data pipeline in sequence:
  1. feature_engineering.py  →  data/features.csv
  2. train_model.py           →  models/*.pkl + data/predictions.csv
  3. explain.py               →  data/shap/*.png + data/shap/shap_values.csv

Usage:
  python run_pipeline.py             # full run
  python run_pipeline.py --skip-shap # skip SHAP (faster)

Note: honeypot.py + simulate_traffic.py must have been run first
to populate the SQLite database.
"""

import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("Feature Engineering",  "feature_engineering.py"),
    ("Model Training",        "train_model.py"),
    ("SHAP Explainability",   "explain.py"),
]


def run_step(name: str, script: str) -> bool:
    """Run a pipeline step and return True on success."""
    print(f"\n{'─' * 55}")
    print(f"  ▶  {name}")
    print(f"{'─' * 55}")
    start  = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n✗  {name} FAILED (exit code {result.returncode})")
        return False

    print(f"\n✓  {name} completed in {elapsed:.1f}s")
    return True


def main():
    skip_shap = "--skip-shap" in sys.argv

    print("=" * 55)
    print("  VICTOR — Full Pipeline Run")
    print("=" * 55)

    # Validate all scripts exist before starting
    for _, script in STEPS:
        if not Path(script).exists():
            print(f"ERROR: {script} not found in current directory.")
            sys.exit(1)

    steps = STEPS if not skip_shap else STEPS[:-1]
    total_start = time.time()

    for name, script in steps:
        ok = run_step(name, script)
        if not ok:
            print("\nPipeline aborted. Fix the error above and re-run.")
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{'=' * 55}")
    print(f"  ✓  All steps complete in {total:.1f}s")
    print(f"     Launch dashboard: streamlit run dashboard.py")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
