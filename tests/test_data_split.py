from __future__ import annotations
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_pipeline.config import SPLIT_COUNTS
from final_pipeline.src.data_split import make_manifest
from final_pipeline.src.generate_data import generate_data

class DataSplitTests(unittest.TestCase):
    def test_grouped_split_is_deterministic_and_exact(self) -> None:
        shipments, _, snapshots = generate_data()
        manifest = make_manifest(shipments)
        self.assertEqual(manifest.split.value_counts().to_dict(), {"train":175,"test":38,"validation":37})
        self.assertEqual(manifest.split.value_counts().to_dict(), SPLIT_COUNTS)
        merged = snapshots.merge(manifest, on="shipment_id")
        self.assertFalse(merged.groupby("shipment_id").split.nunique().gt(1).any())
