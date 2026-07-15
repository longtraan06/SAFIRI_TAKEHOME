from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RISK_THRESHOLD, SPLIT_COUNTS
from src.orchestrator import run


class EvaluationReportingTests(unittest.TestCase):
    def test_full_contract_outputs_validation_final_reports_and_clean_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unrelated.txt").write_text("keep")
            summary = run(root, clean=True)
            required = {
                "01_data_quality/data_quality_report.md",
                "02_split/split_manifest.csv",
                "03_eda/eda_summary.md",
                "04_model_validation/baselines/baseline_metrics_validation.csv",
                "04_model_validation/eta/eta_validation_metrics.csv",
                "04_model_validation/eta/eta_validation_predictions.csv",
                "04_model_validation/risk/risk_validation_metrics.csv",
                "04_model_validation/risk/risk_validation_predictions.csv",
                "05_final_evaluation/metrics/final_test_model_comparison.csv",
                "05_final_evaluation/predictions/final_test_predictions.csv",
                "05_final_evaluation/metrics/final_test_risk_metrics.csv",
                "05_final_evaluation/metrics/final_test_risk_calibration.csv",
                "05_final_evaluation/predictions/final_test_risk_predictions.csv",
                "05_final_evaluation/reports/final_case_studies.md",
                "05_final_evaluation/reports/FINAL_PIPELINE_REPORT.md",
            }
            self.assertTrue(
                all((root / "outputs" / path).exists() for path in required)
            )
            self.assertTrue((root / "outputs" / "03_eda" / "figures").is_dir())
            self.assertTrue(
                (root / "outputs" / "05_final_evaluation" / "figures").is_dir()
            )
            self.assertEqual((root / "unrelated.txt").read_text(), "keep")
            manifest = pd.read_csv(root / "outputs" / "02_split" / "split_manifest.csv")
            self.assertEqual(manifest.split.value_counts().to_dict(), SPLIT_COUNTS)
            risk = pd.read_csv(
                root
                / "outputs"
                / "05_final_evaluation"
                / "predictions"
                / "final_test_risk_predictions.csv"
            )
            self.assertTrue(
                (
                    risk.predicted_material_delay
                    == risk.risk_probability.ge(RISK_THRESHOLD)
                ).all()
            )
            self.assertTrue(risk.risk_probability.between(0, 1).all())
            calibration = pd.read_csv(
                root
                / "outputs"
                / "05_final_evaluation"
                / "metrics"
                / "final_test_risk_calibration.csv"
            )
            self.assertEqual(int(calibration.n.sum()), len(risk))
            report = (
                root
                / "outputs"
                / "05_final_evaluation"
                / "reports"
                / "FINAL_PIPELINE_REPORT.md"
            ).read_text()
            self.assertIn("## 10. Limitations And Reproduction Scope", report)
            self.assertIn("not a new blind or independent evaluation", report)
            self.assertIn("eta_mae_hours", summary)
            self.assertIn("report_paths", summary)
