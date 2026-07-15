from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import OOF_FOLDS, SEED

CATEGORICAL = ["route_id", "carrier", "snapshot_stage"]
NUMERIC = [
    "planned_remaining_hours",
    "calendar_day_of_week",
    "observed_departure_delay_hours",
    "observed_port_arrival_delay_hours",
    "observed_customs_delay_hours",
    "truck_availability_score",
    "congestion_score",
    "weather_severity",
    "document_readiness_score",
    "event_completeness_score",
]


def features(rows: pd.DataFrame) -> pd.DataFrame:
    start_columns = {
        "ORIGIN_DEPARTED": "scheduled_departure_at",
        "PORT_ARRIVED": "scheduled_port_arrival_at",
        "CUSTOMS_CLEARED": "scheduled_customs_clearance_at",
    }
    scheduled_final = pd.to_datetime(rows.scheduled_final_eta, utc=True)
    start = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
    calendar = start.copy()
    for stage, column in start_columns.items():
        mask = rows.snapshot_stage.eq(stage)
        start.loc[mask] = pd.to_datetime(rows.loc[mask, column], utc=True)
        calendar.loc[mask] = pd.to_datetime(
            rows.loc[
                mask,
                (
                    "scheduled_departure_at"
                    if stage == "ORIGIN_DEPARTED"
                    else "scheduled_port_arrival_at"
                ),
            ],
            utc=True,
        )
    result = pd.DataFrame(index=rows.index)
    result["route_id"] = rows.route.astype("string")
    result["carrier"] = rows.carrier.astype("string")
    result["snapshot_stage"] = rows.snapshot_stage.astype("string")
    result["planned_remaining_hours"] = (
        scheduled_final - start
    ).dt.total_seconds() / 3600
    result["calendar_day_of_week"] = calendar.dt.dayofweek
    for column in NUMERIC[2:]:
        result[column] = pd.to_numeric(rows[column], errors="coerce")
    return result[CATEGORICAL + NUMERIC]


def folds(ids: pd.Series) -> dict[str, int]:
    return {
        value: index % OOF_FOLDS
        for index, value in enumerate(
            np.random.default_rng(SEED).permutation(
                np.array(sorted(ids.astype(str).unique()))
            )
        )
    }


def typical_fit(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    latest = pd.Series(
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
    )
    return (
        pd.DataFrame(
            {"route": rows.route, "stage": rows.snapshot_stage, "latest": latest}
        )
        .groupby(["route", "stage"])
        .latest.median(),
        pd.DataFrame({"stage": rows.snapshot_stage, "latest": latest})
        .groupby("stage")
        .latest.median(),
    )


def map_typical(
    rows: pd.DataFrame, route_stage: pd.Series, stage: pd.Series
) -> pd.Series:
    latest = pd.Series(
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
    )
    lookup = route_stage.reindex(
        pd.MultiIndex.from_frame(rows[["route", "snapshot_stage"]])
    ).to_numpy()
    typical = pd.Series(lookup, index=rows.index).fillna(rows.snapshot_stage.map(stage))
    return latest - typical


def oof_history(rows: pd.DataFrame, shipments: pd.DataFrame, mapper) -> pd.Series:
    assigned = shipments.shipment_id.map(folds(shipments.shipment_id))
    result = pd.Series(index=rows.index, dtype=float)
    for fold in range(OOF_FOLDS):
        reference = shipments.loc[assigned.ne(fold)]
        held = set(shipments.loc[assigned.eq(fold), "shipment_id"])
        result.loc[rows.shipment_id.isin(held)] = mapper(
            reference, rows.loc[rows.shipment_id.isin(held)]
        )
    return result


def oof_typical(rows: pd.DataFrame, shipments: pd.DataFrame) -> pd.Series:
    assigned = shipments.shipment_id.map(folds(shipments.shipment_id))
    result = pd.Series(index=rows.index, dtype=float)
    for fold in range(OOF_FOLDS):
        ids = set(shipments.loc[assigned.ne(fold), "shipment_id"])
        held = set(shipments.loc[assigned.eq(fold), "shipment_id"])
        route_stage, stage = typical_fit(rows.loc[rows.shipment_id.isin(ids)])
        result.loc[rows.shipment_id.isin(held)] = map_typical(
            rows.loc[rows.shipment_id.isin(held)], route_stage, stage
        )
    return result


def v2(
    rows: pd.DataFrame, typical: pd.Series, prior: pd.Series | None = None
) -> pd.DataFrame:
    result = features(rows).copy()
    result["port_delay_x_congestion"] = (
        result.observed_port_arrival_delay_hours * result.congestion_score
    )
    result["port_delay_x_document_gap"] = result.observed_port_arrival_delay_hours * (
        1 - result.document_readiness_score
    )
    result["customs_delay_x_truck_shortage"] = result.observed_customs_delay_hours * (
        1 - result.truck_availability_score
    )
    result["observed_delay_vs_route_typical"] = typical.reindex(result.index)
    if prior is not None:
        result["route_prior_final_delay"] = prior.reindex(result.index)
    return result


def preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = [x for x in columns if x not in CATEGORICAL]
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        verbose_feature_names_out=False,
    )
