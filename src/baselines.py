from __future__ import annotations

import pandas as pd


def route_median(shipments: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    values = shipments.groupby("route").final_delay_hours.median()
    return (
        rows.route.map(values)
        .fillna(float(shipments.final_delay_hours.median()))
        .astype(float)
    )
