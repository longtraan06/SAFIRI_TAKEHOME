from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_split import make_manifest
from src.eta_models import fit_eta
from src.generate_data import generate_data


class EtaModelTests(unittest.TestCase):
    def test_stage_routing_uses_frozen_eta_models(self) -> None:
        shipments, _, snapshots = generate_data()
        manifest = make_manifest(shipments)
        train_ids = set(manifest.loc[manifest.split.eq("train"), "shipment_id"])
        validation_ids = set(
            manifest.loc[manifest.split.eq("validation"), "shipment_id"]
        )
        _, predicted = fit_eta(
            snapshots.loc[snapshots.shipment_id.isin(train_ids)],
            shipments.loc[shipments.shipment_id.isin(train_ids)],
            snapshots.loc[snapshots.shipment_id.isin(validation_ids)],
        )
        self.assertTrue(
            (
                predicted.loc[
                    predicted.snapshot_stage.eq("ORIGIN_DEPARTED"), "selected_eta_model"
                ]
                == "Direct HGB v2"
            ).all()
        )
        self.assertTrue(
            (
                predicted.loc[
                    ~predicted.snapshot_stage.eq("ORIGIN_DEPARTED"),
                    "selected_eta_model",
                ]
                == "Structured HGB v2"
            ).all()
        )
        arithmetic = (
            pd.to_datetime(predicted.predicted_final_eta, utc=True)
            - pd.to_datetime(predicted.scheduled_final_eta, utc=True)
        ).dt.total_seconds() / 3600
        self.assertTrue(
            np.allclose(
                arithmetic, predicted.predicted_final_delay_hours, atol=1 / 3600
            )
        )
