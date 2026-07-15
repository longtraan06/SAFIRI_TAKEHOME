from __future__ import annotations
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from final_pipeline.config import N_SHIPMENTS, SEED
from final_pipeline.src.generate_data import generate_data

class GenerateDataTests(unittest.TestCase):
    def test_seeded_generator_is_repeatable_and_complete(self) -> None:
        first = generate_data()
        second = generate_data()
        self.assertEqual(SEED, 20260715)
        self.assertEqual(len(first[0]), N_SHIPMENTS)
        self.assertEqual(len(first[1]), N_SHIPMENTS * 5)
        self.assertTrue(first[0].equals(second[0]))
        self.assertTrue(first[2].equals(second[2]))
