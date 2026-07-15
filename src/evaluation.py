from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from config import RISK_THRESHOLD, STAGES
from src.baselines import route_median


def regression_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "B0 Scheduled ETA": pd.Series(0.0, index=rows.index),
        "B1 Route median": route_median(rows.attrs["train_shipments"], rows),
        "B2 Latest observed carry-forward": pd.Series(
            np.select(
                [
                    rows.snapshot_stage.eq("ORIGIN_DEPARTED"),
                    rows.snapshot_stage.eq("PORT_ARRIVED"),
                ],
                [
                    pd.to_numeric(rows.observed_departure_delay_hours),
                    pd.to_numeric(rows.observed_port_arrival_delay_hours),
                ],
                default=pd.to_numeric(rows.observed_customs_delay_hours),
            ),
            index=rows.index,
        ),
        "Direct HGB v2": rows.direct_v2_predicted_final_delay_hours,
        "Structured HGB v2": rows.structured_v2_predicted_final_delay_hours,
        "Stage-routed v2 policy": rows.predicted_final_delay_hours,
    }
    output = []
    for name, prediction in methods.items():
        for scope in ("ALL", *STAGES):
            mask = (
                prediction.notna()
                if scope == "ALL"
                else prediction.notna() & rows.snapshot_stage.eq(scope)
            )
            if not mask.any():
                continue
            actual = rows.loc[mask, "target_final_delay_hours"]
            output.append(
                dict(
                    method=name,
                    scope=scope,
                    n_snapshots=int(mask.sum()),
                    mae_hours=mean_absolute_error(actual, prediction[mask]),
                    rmse_hours=mean_squared_error(actual, prediction[mask]) ** 0.5,
                )
            )
    return pd.DataFrame(output)


def risk_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    labels = rows.target_is_materially_delayed.astype(int)
    for scope in ("ALL", *STAGES):
        mask = (
            np.ones(len(rows), dtype=bool)
            if scope == "ALL"
            else rows.snapshot_stage.eq(scope)
        )
        y = labels.loc[mask]
        prob = rows.loc[mask, "risk_probability"]
        pred = prob.ge(RISK_THRESHOLD)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        output.append(
            dict(
                method="Risk HGB v2 Stack",
                threshold=RISK_THRESHOLD,
                n_snapshots=int(mask.sum()),
                precision=precision,
                recall=recall,
                f1=f1,
                pr_auc=average_precision_score(y, prob),
                brier_score=brier_score_loss(y, prob),
                scope=scope,
            )
        )
    return pd.DataFrame(output)


def calibration_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise calibrated held-out probabilities without changing the threshold."""
    bins = pd.cut(rows.risk_probability, np.linspace(0, 1, 6), include_lowest=True)
    return (
        pd.DataFrame(
            {
                "bin": bins,
                "probability": rows.risk_probability,
                "label": rows.target_is_materially_delayed,
            }
        )
        .groupby("bin", observed=False)
        .agg(
            n=("label", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_material_delay_rate=("label", "mean"),
        )
        .reset_index()
        .assign(bin=lambda x: x.bin.astype(str))
    )
