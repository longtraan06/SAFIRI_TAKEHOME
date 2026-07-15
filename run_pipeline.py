from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.orchestrator import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    summary = run(ROOT, clean=args.clean)
    if args.run_tests:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(ROOT / "tests"),
                "-v",
            ],
            check=True,
        )
    print("\nFinal pipeline completed.")
    print(f"Reproduction evaluation: {summary['test_snapshots']} held-out snapshots.")
    print(
        "ETA stage-routed v2: "
        f"MAE {summary['eta_mae_hours']:.3f}h | RMSE {summary['eta_rmse_hours']:.3f}h"
    )
    print(
        "Delay Risk: Risk HGB v2 Stack | Platt calibration | "
        f"threshold {summary['risk_threshold']:.2f} | PR-AUC {summary['risk_pr_auc']:.3f} | "
        f"Brier {summary['risk_brier_score']:.3f} | F1 {summary['risk_f1']:.3f}"
    )
    print("Read next:")
    for label, relative_path in summary["report_paths"].items():
        report_path = (ROOT / "outputs" / relative_path).resolve()
        try:
            display_path = report_path.relative_to(Path.cwd().resolve())
        except ValueError:
            display_path = report_path
        print(f"  - {label}: {display_path}")


if __name__ == "__main__":
    main()
