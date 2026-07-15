from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_pipeline.config import OOF_FOLDS, RISK_THRESHOLD
from final_pipeline.pipeline import DIRECT_CONFIG, STRUCTURED_CONFIG, eta_oof, oof_history


class ReproducibilityTests(unittest.TestCase):
    def test_oof_route_rate_excludes_own_shipment_label(self) -> None:
        ids = pd.Series([f"SHP-{index:02d}" for index in range(10)])
        shipments = pd.DataFrame({"shipment_id": ids, "route": ["A", "B"] * 5, "final_delay_hours": [0., 20.] * 5})
        rows = pd.DataFrame({"shipment_id": ids, "route": ["A", "B"] * 5})
        mapper = lambda reference, held: held.route.map(reference.final_delay_hours.gt(12).groupby(reference.route).mean()).fillna(0.)
        before = oof_history(rows, shipments, mapper)
        changed = shipments.copy(); changed.loc[changed.shipment_id.eq("SHP-00"), "final_delay_hours"] = 99.
        after = oof_history(rows, changed, mapper)
        self.assertEqual(before.loc[rows.shipment_id.eq("SHP-00")].iloc[0], after.loc[rows.shipment_id.eq("SHP-00")].iloc[0])

    def test_oof_eta_never_fits_a_holdout_shipment(self) -> None:
        ids = pd.Series([f"SHP-{index:02d}" for index in range(10)])
        shipments = pd.DataFrame({"shipment_id": ids})
        rows = pd.DataFrame({"shipment_id": ids, "snapshot_id": ids, "snapshot_stage": ["ORIGIN_DEPARTED"] * 10})
        calls: list[tuple[set[str], set[str]]] = []
        def spy(fit_rows: pd.DataFrame, fit_shipments: pd.DataFrame, held_rows: pd.DataFrame):
            calls.append((set(fit_shipments.shipment_id), set(held_rows.shipment_id)))
            return {}, pd.DataFrame({"predicted_final_delay_hours": np.zeros(len(held_rows))}, index=held_rows.index)
        with patch("final_pipeline.pipeline.fit_eta", side_effect=spy):
            result = eta_oof(rows, shipments)
        self.assertEqual(len(calls), OOF_FOLDS)
        self.assertTrue(all(not fit_ids.intersection(held_ids) for fit_ids, held_ids in calls))
        self.assertTrue(result.notna().all())

    def test_frozen_hgb_configuration_and_threshold(self) -> None:
        self.assertFalse(DIRECT_CONFIG["early_stopping"])
        self.assertFalse(STRUCTURED_CONFIG["early_stopping"])
        self.assertEqual(RISK_THRESHOLD, .29)
