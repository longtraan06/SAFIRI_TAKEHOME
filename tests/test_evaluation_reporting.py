from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_pipeline.config import RISK_THRESHOLD, SPLIT_COUNTS
from final_pipeline.pipeline import run

class EvaluationReportingTests(unittest.TestCase):
    def test_full_contract_outputs_validation_final_reports_and_clean_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "unrelated.txt").write_text("keep")
            run(root, clean=True)
            required = {"data_quality_report.md", "split_manifest.csv", "eda_summary.md", "baseline_metrics_validation.csv", "eta_validation_metrics.csv", "eta_validation_predictions.csv", "risk_validation_metrics.csv", "risk_validation_predictions.csv", "final_test_model_comparison.csv", "final_test_predictions.csv", "final_test_risk_metrics.csv", "final_test_risk_calibration.csv", "final_test_risk_predictions.csv", "final_case_studies.md", "FINAL_PIPELINE_REPORT.md"}
            self.assertTrue(required.issubset({path.name for path in (root / "outputs").iterdir()}))
            self.assertTrue((root / "outputs" / "figures").is_dir())
            self.assertEqual((root / "unrelated.txt").read_text(), "keep")
            manifest = pd.read_csv(root / "outputs" / "split_manifest.csv")
            self.assertEqual(manifest.split.value_counts().to_dict(), SPLIT_COUNTS)
            risk = pd.read_csv(root / "outputs" / "final_test_risk_predictions.csv")
            self.assertTrue((risk.predicted_material_delay == risk.risk_probability.ge(RISK_THRESHOLD)).all())
            self.assertTrue(risk.risk_probability.between(0, 1).all())
            calibration = pd.read_csv(root / "outputs" / "final_test_risk_calibration.csv")
            self.assertEqual(int(calibration.n.sum()), len(risk))
            report = (root / "outputs" / "FINAL_PIPELINE_REPORT.md").read_text()
            self.assertIn("## 10. Limitations And Reproduction Scope", report)
            self.assertIn("not a new blind or independent evaluation", report)
