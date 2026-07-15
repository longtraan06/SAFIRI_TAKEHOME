from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import MATERIAL_DELAY_HOURS, OOF_FOLDS, RISK_THRESHOLD, SEED
from src.eta_models import STRUCTURED_CONFIG, eta_oof
from src.feature_engineering import (
    folds,
    map_typical,
    oof_history,
    oof_typical,
    preprocessor,
    typical_fit,
    v2,
)


def classifier(columns: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", preprocessor(columns)),
            ("model", HistGradientBoostingClassifier(**STRUCTURED_CONFIG)),
        ]
    )


def risk_features(
    rows: pd.DataFrame, typical: pd.Series, rate: pd.Series, eta: pd.Series
) -> pd.DataFrame:
    result = v2(rows, typical).rename(
        columns={"calendar_day_of_week": "arrival_day_of_week"}
    )
    result["route_material_delay_rate"] = rate.reindex(result.index)
    result["stage_routed_predicted_final_delay_hours"] = eta.reindex(result.index)
    result["delay_margin_to_material_threshold"] = (
        result.stage_routed_predicted_final_delay_hours - MATERIAL_DELAY_HOURS
    )
    return result[
        [
            "route_id",
            "carrier",
            "snapshot_stage",
            "planned_remaining_hours",
            "arrival_day_of_week",
            "observed_departure_delay_hours",
            "observed_port_arrival_delay_hours",
            "observed_customs_delay_hours",
            "congestion_score",
            "weather_severity",
            "document_readiness_score",
            "truck_availability_score",
            "event_completeness_score",
            "port_delay_x_congestion",
            "port_delay_x_document_gap",
            "customs_delay_x_truck_shortage",
            "observed_delay_vs_route_typical",
            "route_material_delay_rate",
            "stage_routed_predicted_final_delay_hours",
            "delay_margin_to_material_threshold",
        ]
    ]


def fit_risk_stack(
    train_rows: pd.DataFrame,
    train_shipments: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    prediction_eta: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit frozen Risk HGB v2 Stack with only group-safe OOF fitting features."""
    train_typical = oof_typical(train_rows, train_shipments)
    route_stage, stage = typical_fit(train_rows)
    prediction_typical = map_typical(prediction_rows, route_stage, stage)
    rate_oof = oof_history(
        train_rows,
        train_shipments,
        lambda s, r: r.route.map(
            (s.final_delay_hours.gt(MATERIAL_DELAY_HOURS).groupby(s.route).mean())
        ).fillna(float(s.final_delay_hours.gt(MATERIAL_DELAY_HOURS).mean())),
    )
    route_rates = (
        train_shipments.final_delay_hours.gt(MATERIAL_DELAY_HOURS)
        .groupby(train_shipments.route)
        .mean()
    )
    prediction_rates = prediction_rows.route.map(route_rates).fillna(
        float(train_shipments.final_delay_hours.gt(MATERIAL_DELAY_HOURS).mean())
    )
    eta_train_oof = eta_oof(train_rows, train_shipments)
    risk_train = risk_features(train_rows, train_typical, rate_oof, eta_train_oof)
    labels = train_rows.target_is_materially_delayed.astype(int)
    assigned = train_rows.shipment_id.map(folds(train_shipments.shipment_id))
    raw_oof = pd.Series(index=train_rows.index, dtype=float)
    for fold in range(OOF_FOLDS):
        model = classifier(risk_train.columns.tolist()).fit(
            risk_train.loc[assigned.ne(fold)], labels.loc[assigned.ne(fold)]
        )
        raw_oof.loc[assigned.eq(fold)] = model.predict_proba(
            risk_train.loc[assigned.eq(fold)]
        )[:, 1]
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=SEED).fit(
        raw_oof.to_numpy().reshape(-1, 1), labels.to_numpy()
    )
    risk_model = classifier(risk_train.columns.tolist()).fit(risk_train, labels)
    result = prediction_eta.copy()
    risk_prediction = risk_features(
        prediction_rows,
        prediction_typical,
        prediction_rates,
        result.predicted_final_delay_hours,
    )
    raw = risk_model.predict_proba(risk_prediction)[:, 1]
    result["risk_raw_probability"] = raw
    result["risk_probability"] = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    result["risk_level"] = pd.cut(
        result.risk_probability,
        [-0.01, 0.35, 0.65, 1],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    result["predicted_material_delay"] = result.risk_probability.ge(RISK_THRESHOLD)
    return {
        "risk_model": risk_model,
        "calibrator": calibrator,
        "raw_oof_probabilities": raw_oof,
    }, result
