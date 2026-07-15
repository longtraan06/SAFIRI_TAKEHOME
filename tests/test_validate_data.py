from __future__ import annotations
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate_data import generate_data
from src.validate_data import validate


class ValidateDataTests(unittest.TestCase):
    def test_valid_generated_files_pass(self) -> None:
        validate(*generate_data())

    def test_duplicate_snapshot_id_is_rejected(self) -> None:
        shipments, events, snapshots = generate_data()
        snapshots.loc[snapshots.index[1], "snapshot_id"] = snapshots.iloc[0].snapshot_id
        with self.assertRaises(ValueError):
            validate(shipments, events, snapshots)

    def test_unknown_reference_and_early_report_are_rejected(self) -> None:
        shipments, events, snapshots = generate_data()
        bad_reference = events.copy()
        bad_reference.loc[0, "shipment_id"] = "UNKNOWN"
        with self.assertRaises(ValueError):
            validate(shipments, bad_reference, snapshots)
        bad_report = events.copy()
        available = bad_report.index[~bad_report["is_update_missing"]][0]
        bad_report.loc[available, "reported_at"] = "2020-01-01T00:00:00Z"
        with self.assertRaises(ValueError):
            validate(shipments, bad_report, snapshots)
