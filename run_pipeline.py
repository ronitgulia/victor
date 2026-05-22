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
from logger import get_logger

logger = get_logger(__name__)


STEPS = [
    ("Feature Engineering",  "feature_engineering.py"),
    ("Model Training",        "train_model.py"),
    ("SHAP Explainability",   "explain.py"),
]


def run_step(name: str, script: str) -> bool:
    """Run a pipeline step and return True on success."""
    logger.info(f"\n{'─' * 55}")
    logger.info(f"  ▶  {name}")
    logger.info(f"{'─' * 55}")
    start  = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode != 0:
        logger.info(f"\n✗  {name} FAILED (exit code {result.returncode})")
        return False

    logger.info(f"\n✓  {name} completed in {elapsed:.1f}s")
    return True


def main():
    skip_shap = "--skip-shap" in sys.argv

    logger.info("=" * 55)
    logger.info("  VICTOR — Full Pipeline Run")
    logger.info("=" * 55)

    # Validate all scripts exist before starting
    for _, script in STEPS:
        if not Path(script).exists():
            logger.error(f"ERROR: {script} not found in current directory.")
            sys.exit(1)

    steps = STEPS if not skip_shap else STEPS[:-1]
    total_start = time.time()

    for name, script in steps:
        ok = run_step(name, script)
        if not ok:
            logger.info("\nPipeline aborted. Fix the error above and re-run.")
            sys.exit(1)

    total = time.time() - total_start
    logger.info(f"\n{'=' * 55}")
    logger.info(f"  ✓  All steps complete in {total:.1f}s")
    logger.info(f"     Launch dashboard: streamlit run dashboard.py")
    logger.info(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
