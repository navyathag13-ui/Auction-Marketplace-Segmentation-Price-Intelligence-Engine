"""
Master pipeline runner — executes all steps in order.
Usage: python scripts/run_all.py
"""

import os
import sys
import subprocess
import time

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCRIPTS  = os.path.join(BASE_DIR, "scripts")
NOTEBOOKS = os.path.join(BASE_DIR, "notebooks")

STEPS = [
    ("Generate synthetic data",  [sys.executable, os.path.join(SCRIPTS, "generate_data.py")]),
    ("Preprocess & clean data",  [sys.executable, os.path.join(SCRIPTS, "preprocess.py")]),
    ("Feature engineering",      [sys.executable, os.path.join(SCRIPTS, "feature_engineering.py")]),
    ("Train segmentation model", [sys.executable, os.path.join(SCRIPTS, "train_segmentation.py")]),
    ("Train price model",        [sys.executable, os.path.join(SCRIPTS, "train_model.py")]),
    ("Initialize database",      [sys.executable, os.path.join(SCRIPTS, "init_db.py")]),
    ("Generate notebooks",       [sys.executable, os.path.join(NOTEBOOKS, "create_notebooks.py")]),
]


def run_step(name: str, cmd: list):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ FAILED: {name} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"  ✓ Done in {elapsed:.1f}s")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Auction Marketplace Segmentation & Price Intelligence   ║")
    print("║  Full Pipeline Runner                                    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    total_start = time.time()
    for name, cmd in STEPS:
        run_step(name, cmd)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE  ({total:.1f}s total)")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  Run API:   uvicorn app.main:app --reload")
    print("  Run tests: pytest tests/ -v")
    print("  Docs:      http://localhost:8000/docs")


if __name__ == "__main__":
    main()
