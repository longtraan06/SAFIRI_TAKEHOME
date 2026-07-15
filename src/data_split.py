from __future__ import annotations

import numpy as np
import pandas as pd

from config import SEED, SPLIT_COUNTS


def make_manifest(shipments: pd.DataFrame) -> pd.DataFrame:
    ids = np.random.default_rng(SEED).permutation(
        np.array(sorted(shipments.shipment_id))
    )
    labels = (
        ["train"] * SPLIT_COUNTS["train"]
        + ["validation"] * SPLIT_COUNTS["validation"]
        + ["test"] * SPLIT_COUNTS["test"]
    )
    return (
        pd.DataFrame({"shipment_id": ids, "split": labels, "split_seed": SEED})
        .sort_values("shipment_id")
        .reset_index(drop=True)
    )
