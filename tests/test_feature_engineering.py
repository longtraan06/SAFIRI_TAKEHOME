from __future__ import annotations
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_pipeline.src.feature_engineering import features, v2
from final_pipeline.src.generate_data import generate_data

class FeatureEngineeringTests(unittest.TestCase):
    def test_features_exclude_targets_and_ids(self) -> None:
        _, _, snapshots = generate_data()
        base = features(snapshots)
        improved = v2(snapshots, base.planned_remaining_hours * 0)
        self.assertNotIn("shipment_id", improved.columns)
        self.assertNotIn("target_final_delay_hours", improved.columns)
        self.assertIn("port_delay_x_congestion", improved.columns)
