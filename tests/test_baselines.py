from __future__ import annotations
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baselines import route_median
from src.generate_data import generate_data


class BaselineTests(unittest.TestCase):
    def test_route_median_uses_training_shipments(self) -> None:
        shipments, _, snapshots = generate_data()
        prediction = route_median(shipments.iloc[:175], snapshots.iloc[:5])
        self.assertEqual(len(prediction), 5)
        self.assertFalse(prediction.isna().any())
