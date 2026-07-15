"""Run the self-contained frozen final pipeline."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

# Support direct execution without reaching outside this package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from final_pipeline.pipeline import run

ROOT = Path(__file__).resolve().parent

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    summary = run(ROOT, clean=args.clean)
    print(f"Completed: {summary['test_snapshots']} held-out snapshots.")
    if args.run_tests:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "tests"), "-v"], check=True)

if __name__ == "__main__": main()
