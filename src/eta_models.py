from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from config import OOF_FOLDS, SEED
from src.baselines import route_median
from src.feature_engineering import (
    folds,
    map_typical,
    oof_history,
    oof_typical,
    preprocessor,
    typical_fit,
    v2,
)

DIRECT_CONFIG = dict(
    learning_rate=0.05,
    max_iter=150,
    max_leaf_nodes=12,
    max_depth=3,
    min_samples_leaf=15,
    l2_regularization=1.5,
    early_stopping=False,
    random_state=SEED,
)
STRUCTURED_CONFIG = dict(
    learning_rate=0.05,
    max_iter=140,
    max_leaf_nodes=10,
    max_depth=3,
    min_samples_leaf=15,
    l2_regularization=2.0,
    early_stopping=False,
    random_state=SEED,
)


def regressor(
    columns: list[str], config: dict[str, Any] = STRUCTURED_CONFIG
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor(columns)),
            ("model", HistGradientBoostingRegressor(**config)),
        ]
    )


def fit_eta(
    train_rows: pd.DataFrame,
    train_shipments: pd.DataFrame,
    prediction_rows: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_prior = oof_history(
        train_rows, train_shipments, lambda s, r: route_median(s, r)
    )
    pred_prior = route_median(train_shipments, prediction_rows)
    train_typical = oof_typical(train_rows, train_shipments)
    route_stage, stage = typical_fit(train_rows)
    pred_typical = map_typical(prediction_rows, route_stage, stage)
    direct_train = v2(train_rows, train_typical, train_prior)
    direct_pred = v2(prediction_rows, pred_typical, pred_prior)
    direct = regressor(direct_train.columns.tolist(), DIRECT_CONFIG).fit(
        direct_train, train_rows.target_final_delay_hours - train_prior
    )
    result = prediction_rows.copy()
    result["direct_v2_predicted_final_delay_hours"] = (
        direct.predict(direct_pred) + pred_prior
    )
    s2train = train_rows.loc[train_rows.snapshot_stage.eq("PORT_ARRIVED")].copy()
    s3train = train_rows.loc[train_rows.snapshot_stage.eq("CUSTOMS_CLEARED")].copy()
    s2pred = prediction_rows.loc[
        prediction_rows.snapshot_stage.eq("PORT_ARRIVED")
    ].copy()
    s3pred = prediction_rows.loc[
        prediction_rows.snapshot_stage.eq("CUSTOMS_CLEARED")
    ].copy()
    truth = train_shipments.set_index("shipment_id")

    def s2_targets(frame: pd.DataFrame) -> pd.DataFrame:
        r = frame.copy()
        snap = pd.to_datetime(r.snapshot_at, utc=True)
        customs = pd.to_datetime(
            r.shipment_id.map(truth.actual_customs_clearance_at), utc=True
        )
        final = pd.to_datetime(
            r.shipment_id.map(truth.actual_final_delivery_at), utc=True
        )
        scheduled_customs = pd.to_datetime(r.scheduled_customs_clearance_at, utc=True)
        scheduled_final = pd.to_datetime(r.scheduled_final_eta, utc=True)
        r["planned_customs_remaining_hours"] = (
            scheduled_customs - snap
        ).dt.total_seconds() / 3600
        r["planned_post_customs_remaining_hours"] = (
            scheduled_final - scheduled_customs
        ).dt.total_seconds() / 3600
        r["customs_deviation_hours"] = (
            customs - snap
        ).dt.total_seconds() / 3600 - r.planned_customs_remaining_hours
        r["post_customs_deviation_hours"] = (
            final - customs
        ).dt.total_seconds() / 3600 - r.planned_post_customs_remaining_hours
        return r

    def s3_targets(frame: pd.DataFrame) -> pd.DataFrame:
        r = frame.copy()
        snap = pd.to_datetime(r.snapshot_at, utc=True)
        final = pd.to_datetime(
            r.shipment_id.map(truth.actual_final_delivery_at), utc=True
        )
        r["planned_inland_remaining_hours"] = (
            pd.to_datetime(r.scheduled_final_eta, utc=True) - snap
        ).dt.total_seconds() / 3600
        r["inland_deviation_hours"] = (
            final - snap
        ).dt.total_seconds() / 3600 - r.planned_inland_remaining_hours
        return r

    s2train = s2_targets(s2train)
    s3train = s3_targets(s3train)
    train_typical_by_id = pd.Series(
        train_typical.to_numpy(), index=train_rows.snapshot_id
    )
    s2f = v2(
        s2train,
        train_typical_by_id.reindex(s2train.snapshot_id).set_axis(s2train.index),
    )
    s3f = v2(
        s3train,
        train_typical_by_id.reindex(s3train.snapshot_id).set_axis(s3train.index),
    )
    s2pf = v2(s2pred, pred_typical.reindex(s2pred.index))
    s3pf = v2(s3pred, pred_typical.reindex(s3pred.index))
    drop = [
        "observed_customs_delay_hours",
        "truck_availability_score",
        "customs_delay_x_truck_shortage",
    ]
    s2f = s2f.drop(columns=drop)
    s2pf = s2pf.drop(columns=drop)
    customs = regressor(s2f.columns.tolist()).fit(s2f, s2train.customs_deviation_hours)
    post = regressor(s2f.columns.tolist()).fit(
        s2f, s2train.post_customs_deviation_hours
    )
    inland = regressor(s3f.columns.tolist()).fit(s3f, s3train.inland_deviation_hours)
    result["structured_v2_predicted_final_delay_hours"] = np.nan
    result["predicted_customs_deviation_hours"] = np.nan
    result["predicted_post_customs_deviation_hours"] = np.nan
    result["predicted_inland_deviation_hours"] = np.nan
    result["planned_customs_remaining_hours"] = np.nan
    result["planned_post_customs_remaining_hours"] = np.nan
    result["planned_inland_remaining_hours"] = np.nan
    if len(s2pred):
        c = customs.predict(s2pf)
        p = post.predict(s2pf)
        plan = s2_targets(s2pred)
        delay = (
            pd.to_datetime(plan.snapshot_at, utc=True)
            + pd.to_timedelta(
                plan.planned_customs_remaining_hours
                + c
                + plan.planned_post_customs_remaining_hours
                + p,
                unit="h",
            )
            - pd.to_datetime(plan.scheduled_final_eta, utc=True)
        ).dt.total_seconds() / 3600
        result.loc[
            s2pred.index,
            [
                "predicted_customs_deviation_hours",
                "predicted_post_customs_deviation_hours",
                "planned_customs_remaining_hours",
                "planned_post_customs_remaining_hours",
                "structured_v2_predicted_final_delay_hours",
            ],
        ] = np.column_stack(
            [
                c,
                p,
                plan.planned_customs_remaining_hours,
                plan.planned_post_customs_remaining_hours,
                delay,
            ]
        )
    if len(s3pred):
        i = inland.predict(s3pf)
        plan = s3_targets(s3pred)
        delay = (
            pd.to_datetime(plan.snapshot_at, utc=True)
            + pd.to_timedelta(plan.planned_inland_remaining_hours + i, unit="h")
            - pd.to_datetime(plan.scheduled_final_eta, utc=True)
        ).dt.total_seconds() / 3600
        result.loc[
            s3pred.index,
            [
                "predicted_inland_deviation_hours",
                "planned_inland_remaining_hours",
                "structured_v2_predicted_final_delay_hours",
            ],
        ] = np.column_stack([i, plan.planned_inland_remaining_hours, delay])
    result["predicted_final_delay_hours"] = np.where(
        result.snapshot_stage.eq("ORIGIN_DEPARTED"),
        result.direct_v2_predicted_final_delay_hours,
        result.structured_v2_predicted_final_delay_hours,
    )
    result["selected_eta_model"] = np.where(
        result.snapshot_stage.eq("ORIGIN_DEPARTED"),
        "Direct HGB v2",
        "Structured HGB v2",
    )
    result["predicted_final_eta"] = (
        pd.to_datetime(result.scheduled_final_eta, utc=True)
        + pd.to_timedelta(result.predicted_final_delay_hours, unit="h")
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "direct": direct,
        "customs": customs,
        "post_customs": post,
        "inland": inland,
    }, result


def eta_oof(rows: pd.DataFrame, shipments: pd.DataFrame) -> pd.Series:
    assignment = shipments.shipment_id.map(folds(shipments.shipment_id))
    output = pd.Series(index=rows.index, dtype=float)
    for fold in range(OOF_FOLDS):
        fit_shipments = shipments.loc[assignment.ne(fold)]
        hold = set(shipments.loc[assignment.eq(fold), "shipment_id"])
        _, predicted = fit_eta(
            rows.loc[rows.shipment_id.isin(set(fit_shipments.shipment_id))],
            fit_shipments,
            rows.loc[rows.shipment_id.isin(hold)],
        )
        output.loc[predicted.index] = predicted.predicted_final_delay_hours
    return output
