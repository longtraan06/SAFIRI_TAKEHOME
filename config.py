from __future__ import annotations

from pathlib import Path

SEED = 20260715
N_SHIPMENTS = 250
SPLIT_COUNTS = {"train": 175, "validation": 37, "test": 38}
OOF_FOLDS = 5
MATERIAL_DELAY_HOURS = 12.0
RISK_THRESHOLD = 0.29
STAGES = ("ORIGIN_DEPARTED", "PORT_ARRIVED", "CUSTOMS_CLEARED")
ETA_POLICY = {
    "ORIGIN_DEPARTED": "Direct HGB v2",
    "PORT_ARRIVED": "Structured HGB v2",
    "CUSTOMS_CLEARED": "Structured HGB v2",
}
PACKAGE_ROOT = Path(__file__).resolve().parent
