from __future__ import annotations
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_pipeline.config import RISK_THRESHOLD
from final_pipeline.src.data_split import make_manifest
from final_pipeline.src.eta_models import fit_eta
from final_pipeline.src.risk_model import risk_features
from final_pipeline.src.generate_data import generate_data
from final_pipeline.pipeline import fit_risk_stack

class RiskModelTests(unittest.TestCase):
    def test_stack_features_include_oof_feature_slots(self) -> None:
        _, _, snapshots = generate_data(); values = snapshots.target_final_delay_hours
        matrix = risk_features(snapshots, values * 0, values.gt(12).astype(float), values)
        self.assertIn("route_material_delay_rate", matrix.columns)
        self.assertIn("stage_routed_predicted_final_delay_hours", matrix.columns)
        self.assertEqual(RISK_THRESHOLD, .29)

    def test_platt_calibrator_receives_complete_oof_raw_probabilities(self) -> None:
        shipments, _, snapshots = generate_data(); manifest = make_manifest(shipments)
        train_ids = set(manifest.loc[manifest.split.eq("train"), "shipment_id"]); validation_ids = set(manifest.loc[manifest.split.eq("validation"), "shipment_id"])
        train_rows = snapshots.loc[snapshots.shipment_id.isin(train_ids)]; validation_rows = snapshots.loc[snapshots.shipment_id.isin(validation_ids)]; train_shipments = shipments.loc[shipments.shipment_id.isin(train_ids)]
        _, validation_eta = fit_eta(train_rows, train_shipments, validation_rows)
        models, predicted = fit_risk_stack(train_rows, train_shipments, validation_rows, validation_eta)
        raw_oof = models["raw_oof_probabilities"]
        self.assertTrue(raw_oof.index.equals(train_rows.index))
        self.assertTrue(raw_oof.between(0, 1).all())
        self.assertTrue(predicted.risk_probability.between(0, 1).all())
