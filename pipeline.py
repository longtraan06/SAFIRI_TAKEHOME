"""Self-contained deterministic final ETA and material-delay pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import MATERIAL_DELAY_HOURS, N_SHIPMENTS, OOF_FOLDS, RISK_THRESHOLD, SEED, SPLIT_COUNTS, STAGES

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORICAL = ["route_id", "carrier", "snapshot_stage"]
NUMERIC = ["planned_remaining_hours", "calendar_day_of_week", "observed_departure_delay_hours", "observed_port_arrival_delay_hours", "observed_customs_delay_hours", "truck_availability_score", "congestion_score", "weather_severity", "document_readiness_score", "event_completeness_score"]
DIRECT_CONFIG = dict(learning_rate=.05, max_iter=150, max_leaf_nodes=12, max_depth=3, min_samples_leaf=15, l2_regularization=1.5, early_stopping=False, random_state=SEED)
STRUCTURED_CONFIG = dict(learning_rate=.05, max_iter=140, max_leaf_nodes=10, max_depth=3, min_samples_leaf=15, l2_regularization=2., early_stopping=False, random_state=SEED)
ROUTES = (
    dict(route="SHANGHAI-SYDNEY", origin="Shanghai, CN", destination_port="Port Botany, AU", final_destination="Sydney, AU", weight=.29, planned_ocean_hours=360., planned_customs_hours=36., planned_inland_hours=30., dispatch_buffer_hours=3., congestion_tendency=.58, ocean_bounds=(300.,450.), customs_bounds=(16.,112.), inland_bounds=(12.,78.)),
    dict(route="SINGAPORE-MELBOURNE", origin="Singapore, SG", destination_port="Melbourne, AU", final_destination="Melbourne, AU", weight=.22, planned_ocean_hours=264., planned_customs_hours=32., planned_inland_hours=27., dispatch_buffer_hours=3., congestion_tendency=.43, ocean_bounds=(215.,350.), customs_bounds=(14.,96.), inland_bounds=(10.,70.)),
    dict(route="HO_CHI_MINH-SYDNEY", origin="Ho Chi Minh City, VN", destination_port="Port Botany, AU", final_destination="Sydney, AU", weight=.24, planned_ocean_hours=312., planned_customs_hours=40., planned_inland_hours=32., dispatch_buffer_hours=3., congestion_tendency=.64, ocean_bounds=(255.,410.), customs_bounds=(18.,124.), inland_bounds=(12.,84.)),
    dict(route="SHENZHEN-BRISBANE", origin="Shenzhen, CN", destination_port="Brisbane, AU", final_destination="Brisbane, AU", weight=.25, planned_ocean_hours=336., planned_customs_hours=34., planned_inland_hours=28., dispatch_buffer_hours=3., congestion_tendency=.49, ocean_bounds=(275.,430.), customs_bounds=(15.,104.), inland_bounds=(11.,74.)),
)
CARRIERS = {"BlueWave Logistics": (.36, -1.5), "Meridian Cargo": (.34, .5), "Pacific Bridge": (.30, 2.2)}
CARGO_RISK = {"general_merchandise": 0., "electronics": 1., "apparel": .5, "industrial_parts": 2.}


def hours(a: datetime, b: datetime) -> float:
    return round((b - a).total_seconds() / 3600, 3)


def iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the frozen synthetic source with its original RNG sequence."""
    rng = np.random.default_rng(SEED)
    shipments: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for number in range(1, N_SHIPMENTS + 1):
        route = ROUTES[int(rng.choice(4, p=[r["weight"] for r in ROUTES]))]
        carrier = tuple(CARRIERS)[int(rng.choice(3, p=[CARRIERS[x][0] for x in CARRIERS]))]
        cargo = str(rng.choice(tuple(CARGO_RISK), p=(.40, .22, .20, .18)))
        scheduled_departure = datetime(2026, 1, 5, 6) + timedelta(hours=int(rng.integers(0, 180 * 24)))
        scheduled_port = scheduled_departure + timedelta(hours=route["planned_ocean_hours"])
        scheduled_customs = scheduled_port + timedelta(hours=route["planned_customs_hours"])
        scheduled_dispatch = scheduled_customs + timedelta(hours=route["dispatch_buffer_hours"])
        scheduled_final = scheduled_port + timedelta(hours=route["planned_customs_hours"] + route["planned_inland_hours"])
        weather = float(np.clip(rng.beta(2., 5.), 0, 1)); congestion = float(np.clip(rng.normal(route["congestion_tendency"], .13), 0, 1))
        documents = float(np.clip(rng.beta(5., 2.2) - .05 * CARGO_RISK[cargo], 0, 1)); truck = float(np.clip(rng.beta(4.3, 2.) - .08 * congestion, 0, 1))
        effect = CARRIERS[carrier][1]
        departure_delay = float(np.clip(rng.normal(weather * 4 + effect - .8, 1.8), -4, 10))
        ocean_increment = float(np.clip(rng.normal(weather * 10 + congestion * 2 + effect + .2 * departure_delay - 3, 3.4), -8, 32))
        actual_departure = scheduled_departure + timedelta(hours=departure_delay); actual_port = actual_departure + timedelta(hours=route["planned_ocean_hours"] + ocean_increment)
        port_delay = hours(scheduled_port, actual_port)
        customs_increment = float(np.clip(rng.normal(congestion * 9 + (1-documents)*12 + max(port_delay, 0)*.18 + CARGO_RISK[cargo] - 5, 4.2), -8, 42))
        actual_customs = actual_port + timedelta(hours=route["planned_customs_hours"] + customs_increment); customs_delay = hours(scheduled_customs, actual_customs)
        dispatch_increment = float(np.clip(rng.normal(max(customs_delay, 0)*.08 + (1-truck)*3 + congestion - 1, 1.5), -2, 12))
        final_increment = float(np.clip(rng.normal(max(customs_delay, 0)*.08 + (1-truck)*8 + weather*2 + dispatch_increment*.1 - 3.5, 2.8), -5, 28))
        actual_dispatch = scheduled_dispatch + timedelta(hours=customs_delay + dispatch_increment)
        actual_final = actual_dispatch + timedelta(hours=route["planned_inland_hours"] - route["dispatch_buffer_hours"] + final_increment)
        final_delay = hours(scheduled_final, actual_final)
        shipment = {**route, "shipment_id": f"SHP-{number:04d}", "carrier": carrier, "cargo_type": cargo, "weather_severity": round(weather,3), "congestion_score": round(congestion,3), "document_readiness_score": round(documents,3), "truck_availability_score": round(truck,3), "scheduled_departure_at": iso(scheduled_departure), "scheduled_port_arrival_at": iso(scheduled_port), "scheduled_customs_clearance_at": iso(scheduled_customs), "scheduled_inland_dispatch_at": iso(scheduled_dispatch), "scheduled_final_eta": iso(scheduled_final), "actual_departure_at": iso(actual_departure), "actual_port_arrival_at": iso(actual_port), "actual_customs_clearance_at": iso(actual_customs), "actual_inland_dispatch_at": iso(actual_dispatch), "actual_final_delivery_at": iso(actual_final), "departure_delay_hours": hours(scheduled_departure, actual_departure), "port_arrival_delay_hours": port_delay, "customs_incremental_delay_hours": round(customs_increment,3), "inland_incremental_delay_hours": round(dispatch_increment + final_increment,3), "final_delay_hours": final_delay, "is_materially_delayed": int(final_delay > MATERIAL_DELAY_HOURS)}
        shipments.append(shipment)
        event_rows = []
        for milestone, scheduled, actual, place in (("ORIGIN_DEPARTED", scheduled_departure, actual_departure, route["origin"]), ("PORT_ARRIVED", scheduled_port, actual_port, route["destination_port"]), ("CUSTOMS_CLEARED", scheduled_customs, actual_customs, route["destination_port"]), ("INLAND_DISPATCHED", scheduled_dispatch, actual_dispatch, route["destination_port"]), ("FINAL_DELIVERED", scheduled_final, actual_final, route["final_destination"])):
            missing = bool(rng.random() < .05); delay = None if missing else float(rng.uniform(4,30) if rng.random() < .14 else rng.uniform(.05,3))
            event_rows.append(dict(shipment_id=shipment["shipment_id"], milestone=milestone, location=place, scheduled_at=iso(scheduled), actual_at=iso(actual), delay_vs_schedule_hours=hours(scheduled,actual), reported_at=None if missing else iso(actual + timedelta(hours=delay)), is_update_missing=missing, is_late_update=False if missing else delay > 4, update_delay_hours=None if missing else round(delay,3)))
        events.extend(event_rows)
        actual_keys = ("actual_departure_at", "actual_port_arrival_at", "actual_customs_clearance_at")
        delay_keys = ("departure_delay_hours", "port_arrival_delay_hours", "customs_delay")
        for index, stage in enumerate(STAGES):
            trigger = event_rows[index]
            if trigger["is_update_missing"]: continue
            available = [event for event in event_rows[:index+1] if not event["is_update_missing"] and event["reported_at"] <= trigger["reported_at"]]
            present = {event["milestone"] for event in available}
            row = dict(snapshot_id=f"{shipment['shipment_id']}-{index+1}", shipment_id=shipment["shipment_id"], snapshot_stage=stage, snapshot_at=trigger["reported_at"], origin=route["origin"], destination_port=route["destination_port"], final_destination=route["final_destination"], route=route["route"], carrier=carrier, cargo_type=cargo, planned_ocean_hours=route["planned_ocean_hours"], planned_customs_hours=route["planned_customs_hours"], planned_inland_hours=route["planned_inland_hours"], scheduled_departure_at=iso(scheduled_departure), scheduled_port_arrival_at=iso(scheduled_port), scheduled_customs_clearance_at=iso(scheduled_customs), scheduled_inland_dispatch_at=iso(scheduled_dispatch), scheduled_final_eta=iso(scheduled_final), weather_severity=round(weather,3), congestion_score=round(congestion,3), document_readiness_score=round(documents,3), observed_departure_delay_hours=np.nan, observed_port_arrival_delay_hours=np.nan, observed_customs_delay_hours=np.nan, truck_availability_score=np.nan, upstream_missing_update_count=index+1-len(available), event_completeness_score=round(len(available)/(index+1),3), target_final_delay_hours=final_delay, target_is_materially_delayed=int(final_delay > MATERIAL_DELAY_HOURS), target_actual_final_delivery_at=iso(actual_final))
            if "ORIGIN_DEPARTED" in present: row["observed_departure_delay_hours"] = shipment["departure_delay_hours"]
            if "PORT_ARRIVED" in present: row["observed_port_arrival_delay_hours"] = shipment["port_arrival_delay_hours"]
            if "CUSTOMS_CLEARED" in present: row["observed_customs_delay_hours"] = customs_delay; row["truck_availability_score"] = shipment["truck_availability_score"]
            snapshots.append(row)
    return pd.DataFrame(shipments), pd.DataFrame(events), pd.DataFrame(snapshots)


def validate(shipments: pd.DataFrame, events: pd.DataFrame, snapshots: pd.DataFrame) -> None:
    if len(shipments) != N_SHIPMENTS or shipments.shipment_id.nunique() != N_SHIPMENTS:
        raise ValueError("Invalid shipment population")
    shipment_ids = set(shipments.shipment_id)
    if not set(events.shipment_id).issubset(shipment_ids) or not set(snapshots.shipment_id).issubset(shipment_ids):
        raise ValueError("Events or snapshots reference an unknown shipment")
    if events.groupby("shipment_id").size().ne(5).any():
        raise ValueError("Every shipment must have five events")
    if not set(snapshots.snapshot_stage).issubset(STAGES):
        raise ValueError("Unsupported snapshot stage")
    if snapshots.snapshot_id.duplicated().any() or snapshots.target_final_delay_hours.isna().any():
        raise ValueError("Invalid snapshots")
    actual_columns = ["actual_departure_at", "actual_port_arrival_at", "actual_customs_clearance_at", "actual_inland_dispatch_at", "actual_final_delivery_at"]
    actual_times = shipments[actual_columns].apply(pd.to_datetime, utc=True)
    if (actual_times.diff(axis=1).iloc[:, 1:] < pd.Timedelta(0)).any().any():
        raise ValueError("Shipment actual milestone chronology is invalid")
    missing = events["is_update_missing"].astype(bool)
    reported = pd.to_datetime(events["reported_at"], utc=True)
    actual = pd.to_datetime(events["actual_at"], utc=True)
    if reported.loc[missing].notna().any() or reported.loc[~missing].isna().any() or (reported.loc[~missing] < actual.loc[~missing]).any():
        raise ValueError("Event reporting-time contract is invalid")
    s1 = snapshots.snapshot_stage.eq("ORIGIN_DEPARTED")
    s2 = snapshots.snapshot_stage.eq("PORT_ARRIVED")
    future_at_s1 = ["observed_port_arrival_delay_hours", "observed_customs_delay_hours", "truck_availability_score"]
    future_at_s2 = ["observed_customs_delay_hours", "truck_availability_score"]
    if snapshots.loc[s1, future_at_s1].notna().any().any() or snapshots.loc[s2, future_at_s2].notna().any().any():
        raise ValueError("Snapshot contains future-stage evidence")


def make_manifest(shipments: pd.DataFrame) -> pd.DataFrame:
    ids = np.random.default_rng(SEED).permutation(np.array(sorted(shipments.shipment_id)))
    labels = ["train"]*SPLIT_COUNTS["train"] + ["validation"]*SPLIT_COUNTS["validation"] + ["test"]*SPLIT_COUNTS["test"]
    return pd.DataFrame({"shipment_id": ids, "split": labels, "split_seed": SEED}).sort_values("shipment_id").reset_index(drop=True)


def features(rows: pd.DataFrame) -> pd.DataFrame:
    start_columns = {"ORIGIN_DEPARTED": "scheduled_departure_at", "PORT_ARRIVED": "scheduled_port_arrival_at", "CUSTOMS_CLEARED": "scheduled_customs_clearance_at"}
    scheduled_final = pd.to_datetime(rows.scheduled_final_eta, utc=True)
    start = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
    calendar = start.copy()
    for stage, column in start_columns.items():
        mask = rows.snapshot_stage.eq(stage); start.loc[mask] = pd.to_datetime(rows.loc[mask, column], utc=True)
        calendar.loc[mask] = pd.to_datetime(rows.loc[mask, "scheduled_departure_at" if stage == "ORIGIN_DEPARTED" else "scheduled_port_arrival_at"], utc=True)
    result = pd.DataFrame(index=rows.index)
    result["route_id"] = rows.route.astype("string"); result["carrier"] = rows.carrier.astype("string"); result["snapshot_stage"] = rows.snapshot_stage.astype("string")
    result["planned_remaining_hours"] = (scheduled_final-start).dt.total_seconds()/3600; result["calendar_day_of_week"] = calendar.dt.dayofweek
    for column in NUMERIC[2:]: result[column] = pd.to_numeric(rows[column], errors="coerce")
    return result[CATEGORICAL + NUMERIC]


def folds(ids: pd.Series) -> dict[str, int]:
    return {value: index % OOF_FOLDS for index, value in enumerate(np.random.default_rng(SEED).permutation(np.array(sorted(ids.astype(str).unique()))))}


def route_median(shipments: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    values = shipments.groupby("route").final_delay_hours.median(); return rows.route.map(values).fillna(float(shipments.final_delay_hours.median())).astype(float)


def typical_fit(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    latest = pd.Series(np.select([rows.snapshot_stage.eq("ORIGIN_DEPARTED"), rows.snapshot_stage.eq("PORT_ARRIVED")], [pd.to_numeric(rows.observed_departure_delay_hours), pd.to_numeric(rows.observed_port_arrival_delay_hours)], default=pd.to_numeric(rows.observed_customs_delay_hours)), index=rows.index)
    return pd.DataFrame({"route":rows.route,"stage":rows.snapshot_stage,"latest":latest}).groupby(["route","stage"]).latest.median(), pd.DataFrame({"stage":rows.snapshot_stage,"latest":latest}).groupby("stage").latest.median()


def map_typical(rows: pd.DataFrame, route_stage: pd.Series, stage: pd.Series) -> pd.Series:
    latest = pd.Series(np.select([rows.snapshot_stage.eq("ORIGIN_DEPARTED"), rows.snapshot_stage.eq("PORT_ARRIVED")], [pd.to_numeric(rows.observed_departure_delay_hours), pd.to_numeric(rows.observed_port_arrival_delay_hours)], default=pd.to_numeric(rows.observed_customs_delay_hours)), index=rows.index)
    lookup = route_stage.reindex(pd.MultiIndex.from_frame(rows[["route","snapshot_stage"]])).to_numpy(); typical = pd.Series(lookup,index=rows.index).fillna(rows.snapshot_stage.map(stage)); return latest - typical


def oof_history(rows: pd.DataFrame, shipments: pd.DataFrame, mapper) -> pd.Series:
    assigned = shipments.shipment_id.map(folds(shipments.shipment_id)); result = pd.Series(index=rows.index,dtype=float)
    for fold in range(OOF_FOLDS):
        reference = shipments.loc[assigned.ne(fold)]; held = set(shipments.loc[assigned.eq(fold),"shipment_id"]); result.loc[rows.shipment_id.isin(held)] = mapper(reference, rows.loc[rows.shipment_id.isin(held)])
    return result


def oof_typical(rows: pd.DataFrame, shipments: pd.DataFrame) -> pd.Series:
    assigned = shipments.shipment_id.map(folds(shipments.shipment_id)); result = pd.Series(index=rows.index,dtype=float)
    for fold in range(OOF_FOLDS):
        ids = set(shipments.loc[assigned.ne(fold),"shipment_id"]); held = set(shipments.loc[assigned.eq(fold),"shipment_id"]); route_stage, stage = typical_fit(rows.loc[rows.shipment_id.isin(ids)]); result.loc[rows.shipment_id.isin(held)] = map_typical(rows.loc[rows.shipment_id.isin(held)], route_stage, stage)
    return result


def v2(rows: pd.DataFrame, typical: pd.Series, prior: pd.Series | None = None) -> pd.DataFrame:
    result = features(rows).copy(); result["port_delay_x_congestion"] = result.observed_port_arrival_delay_hours * result.congestion_score; result["port_delay_x_document_gap"] = result.observed_port_arrival_delay_hours * (1-result.document_readiness_score); result["customs_delay_x_truck_shortage"] = result.observed_customs_delay_hours * (1-result.truck_availability_score); result["observed_delay_vs_route_typical"] = typical.reindex(result.index)
    if prior is not None: result["route_prior_final_delay"] = prior.reindex(result.index)
    return result


def preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = [x for x in columns if x not in CATEGORICAL]
    return ColumnTransformer([("numeric",SimpleImputer(strategy="median",add_indicator=True),numeric),("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),CATEGORICAL)], verbose_feature_names_out=False)


def regressor(columns: list[str], config: dict[str, Any] = STRUCTURED_CONFIG) -> Pipeline: return Pipeline([("preprocess",preprocessor(columns)),("model",HistGradientBoostingRegressor(**config))])
def classifier(columns: list[str]) -> Pipeline: return Pipeline([("preprocess",preprocessor(columns)),("model",HistGradientBoostingClassifier(**STRUCTURED_CONFIG))])


def fit_eta(train_rows: pd.DataFrame, train_shipments: pd.DataFrame, prediction_rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    train_prior = oof_history(train_rows, train_shipments, lambda s,r: route_median(s,r)); pred_prior = route_median(train_shipments,prediction_rows)
    train_typical = oof_typical(train_rows,train_shipments); route_stage, stage = typical_fit(train_rows); pred_typical = map_typical(prediction_rows,route_stage,stage)
    direct_train = v2(train_rows,train_typical,train_prior); direct_pred = v2(prediction_rows,pred_typical,pred_prior)
    direct = regressor(direct_train.columns.tolist(), DIRECT_CONFIG).fit(direct_train,train_rows.target_final_delay_hours-train_prior); result = prediction_rows.copy(); result["direct_v2_predicted_final_delay_hours"] = direct.predict(direct_pred) + pred_prior
    s2train = train_rows.loc[train_rows.snapshot_stage.eq("PORT_ARRIVED")].copy(); s3train = train_rows.loc[train_rows.snapshot_stage.eq("CUSTOMS_CLEARED")].copy(); s2pred = prediction_rows.loc[prediction_rows.snapshot_stage.eq("PORT_ARRIVED")].copy(); s3pred = prediction_rows.loc[prediction_rows.snapshot_stage.eq("CUSTOMS_CLEARED")].copy()
    truth = train_shipments.set_index("shipment_id")
    def s2_targets(frame: pd.DataFrame) -> pd.DataFrame:
        r=frame.copy(); snap=pd.to_datetime(r.snapshot_at,utc=True); customs=pd.to_datetime(r.shipment_id.map(truth.actual_customs_clearance_at),utc=True); final=pd.to_datetime(r.shipment_id.map(truth.actual_final_delivery_at),utc=True); scheduled_customs=pd.to_datetime(r.scheduled_customs_clearance_at,utc=True); scheduled_final=pd.to_datetime(r.scheduled_final_eta,utc=True); r["planned_customs_remaining_hours"]=(scheduled_customs-snap).dt.total_seconds()/3600; r["planned_post_customs_remaining_hours"]=(scheduled_final-scheduled_customs).dt.total_seconds()/3600; r["customs_deviation_hours"]=(customs-snap).dt.total_seconds()/3600-r.planned_customs_remaining_hours; r["post_customs_deviation_hours"]=(final-customs).dt.total_seconds()/3600-r.planned_post_customs_remaining_hours; return r
    def s3_targets(frame: pd.DataFrame) -> pd.DataFrame:
        r=frame.copy(); snap=pd.to_datetime(r.snapshot_at,utc=True); final=pd.to_datetime(r.shipment_id.map(truth.actual_final_delivery_at),utc=True); r["planned_inland_remaining_hours"]=(pd.to_datetime(r.scheduled_final_eta,utc=True)-snap).dt.total_seconds()/3600; r["inland_deviation_hours"]=(final-snap).dt.total_seconds()/3600-r.planned_inland_remaining_hours; return r
    s2train=s2_targets(s2train); s3train=s3_targets(s3train)
    train_typical_by_id=pd.Series(train_typical.to_numpy(),index=train_rows.snapshot_id); s2f=v2(s2train,train_typical_by_id.reindex(s2train.snapshot_id).set_axis(s2train.index)); s3f=v2(s3train,train_typical_by_id.reindex(s3train.snapshot_id).set_axis(s3train.index)); s2pf=v2(s2pred,pred_typical.reindex(s2pred.index)); s3pf=v2(s3pred,pred_typical.reindex(s3pred.index)); drop=["observed_customs_delay_hours","truck_availability_score","customs_delay_x_truck_shortage"]; s2f=s2f.drop(columns=drop); s2pf=s2pf.drop(columns=drop)
    customs=regressor(s2f.columns.tolist()).fit(s2f,s2train.customs_deviation_hours); post=regressor(s2f.columns.tolist()).fit(s2f,s2train.post_customs_deviation_hours); inland=regressor(s3f.columns.tolist()).fit(s3f,s3train.inland_deviation_hours)
    result["structured_v2_predicted_final_delay_hours"] = np.nan; result["predicted_customs_deviation_hours"] = np.nan; result["predicted_post_customs_deviation_hours"] = np.nan; result["predicted_inland_deviation_hours"] = np.nan; result["planned_customs_remaining_hours"] = np.nan; result["planned_post_customs_remaining_hours"] = np.nan; result["planned_inland_remaining_hours"] = np.nan
    if len(s2pred):
        c=customs.predict(s2pf); p=post.predict(s2pf); plan=s2_targets(s2pred); delay=(pd.to_datetime(plan.snapshot_at,utc=True)+pd.to_timedelta(plan.planned_customs_remaining_hours+c+plan.planned_post_customs_remaining_hours+p,unit="h")-pd.to_datetime(plan.scheduled_final_eta,utc=True)).dt.total_seconds()/3600; result.loc[s2pred.index,["predicted_customs_deviation_hours","predicted_post_customs_deviation_hours","planned_customs_remaining_hours","planned_post_customs_remaining_hours","structured_v2_predicted_final_delay_hours"]] = np.column_stack([c,p,plan.planned_customs_remaining_hours,plan.planned_post_customs_remaining_hours,delay])
    if len(s3pred):
        i=inland.predict(s3pf); plan=s3_targets(s3pred); delay=(pd.to_datetime(plan.snapshot_at,utc=True)+pd.to_timedelta(plan.planned_inland_remaining_hours+i,unit="h")-pd.to_datetime(plan.scheduled_final_eta,utc=True)).dt.total_seconds()/3600; result.loc[s3pred.index,["predicted_inland_deviation_hours","planned_inland_remaining_hours","structured_v2_predicted_final_delay_hours"]] = np.column_stack([i,plan.planned_inland_remaining_hours,delay])
    result["predicted_final_delay_hours"] = np.where(result.snapshot_stage.eq("ORIGIN_DEPARTED"),result.direct_v2_predicted_final_delay_hours,result.structured_v2_predicted_final_delay_hours); result["selected_eta_model"] = np.where(result.snapshot_stage.eq("ORIGIN_DEPARTED"),"Direct HGB v2","Structured HGB v2"); result["predicted_final_eta"] = (pd.to_datetime(result.scheduled_final_eta,utc=True)+pd.to_timedelta(result.predicted_final_delay_hours,unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"direct":direct,"customs":customs,"post_customs":post,"inland":inland}, result


def eta_oof(rows: pd.DataFrame, shipments: pd.DataFrame) -> pd.Series:
    assignment=shipments.shipment_id.map(folds(shipments.shipment_id)); output=pd.Series(index=rows.index,dtype=float)
    for fold in range(OOF_FOLDS):
        fit_shipments=shipments.loc[assignment.ne(fold)]; hold=set(shipments.loc[assignment.eq(fold),"shipment_id"]); _, predicted=fit_eta(rows.loc[rows.shipment_id.isin(set(fit_shipments.shipment_id))],fit_shipments,rows.loc[rows.shipment_id.isin(hold)]); output.loc[predicted.index]=predicted.predicted_final_delay_hours
    return output


def risk_features(rows: pd.DataFrame, typical: pd.Series, rate: pd.Series, eta: pd.Series) -> pd.DataFrame:
    result=v2(rows,typical).rename(columns={"calendar_day_of_week":"arrival_day_of_week"}); result["route_material_delay_rate"]=rate.reindex(result.index); result["stage_routed_predicted_final_delay_hours"]=eta.reindex(result.index); result["delay_margin_to_material_threshold"]=result.stage_routed_predicted_final_delay_hours-MATERIAL_DELAY_HOURS
    return result[["route_id","carrier","snapshot_stage","planned_remaining_hours","arrival_day_of_week","observed_departure_delay_hours","observed_port_arrival_delay_hours","observed_customs_delay_hours","congestion_score","weather_severity","document_readiness_score","truck_availability_score","event_completeness_score","port_delay_x_congestion","port_delay_x_document_gap","customs_delay_x_truck_shortage","observed_delay_vs_route_typical","route_material_delay_rate","stage_routed_predicted_final_delay_hours","delay_margin_to_material_threshold"]]


def metric_rows(labels: pd.Series, prediction: pd.Series, method: str) -> list[dict[str,Any]]:
    result=[]
    for stage in ("ALL",*STAGES):
        mask=np.ones(len(labels),dtype=bool) if stage=="ALL" else labels.index.to_series().map(lambda x: False) # replaced by caller
        result.append({"method":method,"scope":stage})
    return result


def regression_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    methods={"B0 Scheduled ETA":pd.Series(0.,index=rows.index),"B1 Route median":route_median(rows.attrs["train_shipments"],rows),"B2 Latest observed carry-forward":pd.Series(np.select([rows.snapshot_stage.eq("ORIGIN_DEPARTED"),rows.snapshot_stage.eq("PORT_ARRIVED")],[pd.to_numeric(rows.observed_departure_delay_hours),pd.to_numeric(rows.observed_port_arrival_delay_hours)],default=pd.to_numeric(rows.observed_customs_delay_hours)),index=rows.index),"Direct HGB v2":rows.direct_v2_predicted_final_delay_hours,"Structured HGB v2":rows.structured_v2_predicted_final_delay_hours,"Stage-routed v2 policy":rows.predicted_final_delay_hours}
    output=[]
    for name,prediction in methods.items():
        for scope in ("ALL",*STAGES):
            mask=prediction.notna() if scope=="ALL" else prediction.notna() & rows.snapshot_stage.eq(scope)
            if not mask.any(): continue
            actual=rows.loc[mask,"target_final_delay_hours"]; output.append(dict(method=name,scope=scope,n_snapshots=int(mask.sum()),mae_hours=mean_absolute_error(actual,prediction[mask]),rmse_hours=mean_squared_error(actual,prediction[mask])**.5))
    return pd.DataFrame(output)


def risk_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    output=[]; labels=rows.target_is_materially_delayed.astype(int)
    for scope in ("ALL",*STAGES):
        mask=np.ones(len(rows),dtype=bool) if scope=="ALL" else rows.snapshot_stage.eq(scope); y=labels.loc[mask]; prob=rows.loc[mask,"risk_probability"]; pred=prob.ge(RISK_THRESHOLD); precision,recall,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0); output.append(dict(method="Risk HGB v2 Stack",threshold=RISK_THRESHOLD,n_snapshots=int(mask.sum()),precision=precision,recall=recall,f1=f1,pr_auc=average_precision_score(y,prob),brier_score=brier_score_loss(y,prob),scope=scope))
    return pd.DataFrame(output)


def calibration_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarise calibrated held-out probabilities without changing the threshold."""
    bins = pd.cut(rows.risk_probability, np.linspace(0, 1, 6), include_lowest=True)
    return pd.DataFrame({"bin": bins, "probability": rows.risk_probability, "label": rows.target_is_materially_delayed}).groupby("bin", observed=False).agg(n=("label", "size"), mean_predicted_probability=("probability", "mean"), observed_material_delay_rate=("label", "mean")).reset_index().assign(bin=lambda x: x.bin.astype(str))


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
    rate_oof = oof_history(train_rows, train_shipments, lambda s, r: r.route.map((s.final_delay_hours.gt(MATERIAL_DELAY_HOURS).groupby(s.route).mean())).fillna(float(s.final_delay_hours.gt(MATERIAL_DELAY_HOURS).mean())))
    route_rates = train_shipments.final_delay_hours.gt(MATERIAL_DELAY_HOURS).groupby(train_shipments.route).mean()
    prediction_rates = prediction_rows.route.map(route_rates).fillna(float(train_shipments.final_delay_hours.gt(MATERIAL_DELAY_HOURS).mean()))
    eta_train_oof = eta_oof(train_rows, train_shipments)
    risk_train = risk_features(train_rows, train_typical, rate_oof, eta_train_oof)
    labels = train_rows.target_is_materially_delayed.astype(int)
    assigned = train_rows.shipment_id.map(folds(train_shipments.shipment_id))
    raw_oof = pd.Series(index=train_rows.index, dtype=float)
    for fold in range(OOF_FOLDS):
        model = classifier(risk_train.columns.tolist()).fit(risk_train.loc[assigned.ne(fold)], labels.loc[assigned.ne(fold)])
        raw_oof.loc[assigned.eq(fold)] = model.predict_proba(risk_train.loc[assigned.eq(fold)])[:, 1]
    calibrator = LogisticRegression(C=1., solver="lbfgs", random_state=SEED).fit(raw_oof.to_numpy().reshape(-1, 1), labels.to_numpy())
    risk_model = classifier(risk_train.columns.tolist()).fit(risk_train, labels)
    result = prediction_eta.copy()
    risk_prediction = risk_features(prediction_rows, prediction_typical, prediction_rates, result.predicted_final_delay_hours)
    raw = risk_model.predict_proba(risk_prediction)[:, 1]
    result["risk_raw_probability"] = raw
    result["risk_probability"] = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    result["risk_level"] = pd.cut(result.risk_probability, [-.01, .35, .65, 1], labels=["LOW", "MEDIUM", "HIGH"])
    result["predicted_material_delay"] = result.risk_probability.ge(RISK_THRESHOLD)
    return {"risk_model": risk_model, "calibrator": calibrator, "raw_oof_probabilities": raw_oof}, result


def write_reports(root: Path, shipments: pd.DataFrame, events: pd.DataFrame, snapshots: pd.DataFrame, manifest: pd.DataFrame, comparison: pd.DataFrame, risk: pd.DataFrame, predictions: pd.DataFrame, validation_comparison: pd.DataFrame, validation_risk: pd.DataFrame) -> None:
    outputs=root/"outputs"; figures=outputs/"figures"; figures.mkdir(parents=True,exist_ok=True)
    (outputs/"data_quality_report.md").write_text(f"# Data Quality\n\n- Shipments: {len(shipments)}\n- Events: {len(shipments)*5}\n- Snapshots: {len(snapshots)}\n- Validation: passed.\n",encoding="utf-8")
    counts = snapshots.snapshot_stage.value_counts().reindex(STAGES, fill_value=0)
    shipment_counts_route = shipments.route.value_counts()
    shipment_counts_carrier = shipments.carrier.value_counts()
    route_duration = shipments.groupby("route")[["planned_ocean_hours", "planned_customs_hours", "planned_inland_hours"]].median()
    route_delay = shipments.assign(material=shipments.final_delay_hours.gt(MATERIAL_DELAY_HOURS)).groupby("route").agg(shipments=("shipment_id", "size"), material_delay_rate=("material", "mean"), median_final_delay_hours=("final_delay_hours", "median"))
    delay_buckets = pd.cut(shipments.final_delay_hours, [-np.inf, 0, 6, 12, np.inf], labels=["early_or_on_time", "1_to_6h", "6_to_12h", "over_12h"]).value_counts().reindex(["early_or_on_time", "1_to_6h", "6_to_12h", "over_12h"], fill_value=0)
    propagation = pd.DataFrame({"port_delay_to_customs_increment": shipments.port_arrival_delay_hours.corr(shipments.customs_incremental_delay_hours), "customs_increment_to_inland_increment": shipments.customs_incremental_delay_hours.corr(shipments.inland_incremental_delay_hours), "port_delay_to_final_delay": shipments.port_arrival_delay_hours.corr(shipments.final_delay_hours)}, index=["correlation"]).T
    update_summary = events.groupby("milestone").agg(events=("shipment_id", "size"), missing_update_rate=("is_update_missing", "mean"), late_update_rate=("is_late_update", "mean"))
    availability = snapshots.groupby("snapshot_stage")[["observed_departure_delay_hours", "observed_port_arrival_delay_hours", "observed_customs_delay_hours", "truck_availability_score"]].apply(lambda x: x.notna().mean()).reindex(STAGES)
    def table(frame: pd.DataFrame) -> str:
        return "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in frame.reset_index().itertuples(index=False, name=None))
    eda = "# EDA Summary\n\n## Shipment Counts By Route\n\n| Route | Shipments |\n| --- | ---: |\n" + table(shipment_counts_route.rename("shipments").to_frame())
    eda += "\n\n## Shipment Counts By Carrier\n\n| Carrier | Shipments |\n| --- | ---: |\n" + table(shipment_counts_carrier.rename("shipments").to_frame())
    eda += "\n\n## Snapshot Counts By Stage\n\n| Stage | Snapshots |\n| --- | ---: |\n" + table(counts.rename("snapshots").to_frame())
    eda += "\n\n## Planned Duration By Route\n\n| Route | Ocean | Customs | Inland |\n| --- | ---: | ---: | ---: |\n" + table(route_duration)
    eda += "\n\n## Final Delay Distribution And Buckets\n\n" + f"Mean={shipments.final_delay_hours.mean():.3f}h; median={shipments.final_delay_hours.median():.3f}h; standard deviation={shipments.final_delay_hours.std():.3f}h.\n\n| Bucket | Shipments |\n| --- | ---: |\n" + table(delay_buckets.rename("shipments").to_frame())
    eda += "\n\n## Route Material Delay Rate And Median\n\n| Route | Shipments | Material delay rate | Median final delay hours |\n| --- | ---: | ---: | ---: |\n" + table(route_delay)
    eda += "\n\n## Delay Propagation Relations\n\n| Relation | Correlation |\n| --- | ---: |\n" + table(propagation)
    eda += "\n\n## Missing And Late Update Summary\n\n| Milestone | Events | Missing rate | Late rate |\n| --- | ---: | ---: | ---: |\n" + table(update_summary)
    eda += "\n\n## Feature Availability By Prediction Stage\n\nValues are fractions available at S1/S2/S3.\n\n| Stage | Departure delay | Port delay | Customs delay | Truck availability |\n| --- | ---: | ---: | ---: | ---: |\n" + table(availability) + "\n"
    (outputs/"eda_summary.md").write_text(eda, encoding="utf-8")
    frozen="# Frozen Policy\n\n- Seed: `20260715`; grouped split: `175/37/38`.\n- ETA: Direct residual HGB v2 at S1; Structured planned-deviation HGB v2 at S2/S3.\n- Risk: Risk HGB v2 Stack using OOF route material-delay rate and OOF stage-routed ETA features; Platt calibration; fixed threshold `0.29`.\n"
    (outputs/"frozen_policy.md").write_text(frozen,encoding="utf-8")
    eta_rows = "\n".join(f"| {r.method} | {r.scope} | {r.n_snapshots} | {r.mae_hours:.3f} | {r.rmse_hours:.3f} |" for _, r in comparison.iterrows())
    risk_rows = "\n".join(f"| {r.method} | {r.scope} | {r.pr_auc:.3f} | {r.brier_score:.3f} | {r.f1:.3f} |" for _, r in risk.iterrows())
    validation_eta_rows = "\n".join(f"| {r.method} | {r.scope} | {r.n_snapshots} | {r.mae_hours:.3f} | {r.rmse_hours:.3f} |" for _, r in validation_comparison.loc[validation_comparison.method.isin(["Direct HGB v2", "Structured HGB v2", "Stage-routed v2 policy"])].iterrows())
    validation_risk_rows = "\n".join(f"| {r.method} | {r.scope} | {r.pr_auc:.3f} | {r.brier_score:.3f} | {r.f1:.3f} |" for _, r in validation_risk.iterrows())
    report = "# Final Pipeline Report\n\n## 1. Reproduction Command\n\n`python final_pipeline/run_pipeline.py --clean --run-tests`\n\n## 2. Seed, Dataset, And Grouped Split\n\nSynthetic data uses seed `20260715`: 250 shipments, five events per shipment, and milestone snapshots. Shipment groups are split train/validation/test as `175/37/38`.\n\n## 3. Data Quality\n\nThe generated dataset passed population, event-count, snapshot-ID, stage, and target-completeness validation. See `data_quality_report.md`.\n\n## 4. EDA Insights\n\n`eda_summary.md` and `figures/` document shipment route/carrier/stage counts, planned durations, final-delay distribution/buckets, route delay rates and medians, propagation correlations, update quality, and stage feature availability.\n\n## 5. Baseline Results\n\nB0 is scheduled ETA, B1 maps a train-fitted route median, and B2 carries forward the latest available observed delay. Train-only validation baseline results are in `baseline_metrics_validation.csv`.\n\n## 6. ETA Architecture\n\nThe frozen ETA policy routes S1 to Direct residual HGB v2 with an OOF route-delay prior. S2/S3 use Structured planned-deviation HGB v2 waterfall components. It is stage routing, not a test-selected ensemble.\n\n## 7. Risk Architecture\n\nRisk HGB v2 Stack uses OOF route material-delay rates and OOF stage-routed ETA features. Platt calibration is fitted only on OOF raw probabilities; the alert threshold is fixed at `0.29`.\n\n## 8. Validation And Test/Reproduction Metrics\n\nTrain-only validation uses train shipments only and is saved in the validation CSVs.\n\n### Train-only Validation ETA\n\n| Method | Scope | n | MAE | RMSE |\n| --- | --- | ---: | ---: | ---: |\n"+validation_eta_rows+"\n\n### Train-only Validation Risk\n\n| Method | Scope | PR-AUC | Brier | F1 |\n| --- | --- | ---: | ---: | ---: |\n"+validation_risk_rows+"\n\nThe following final test rerun is reproducibility verification of the frozen synthetic benchmark, **not a new blind or independent evaluation**.\n\n### Final ETA\n\n| Method | Scope | n | MAE | RMSE |\n| --- | --- | ---: | ---: | ---: |\n"+eta_rows+"\n\n### Final Risk\n\n| Method | Scope | PR-AUC | Brier | F1 |\n| --- | --- | ---: | ---: | ---: |\n"+risk_rows+"\n\n## 9. Leakage Safeguards\n\nSplits are shipment-grouped. All validation fits use train shipments only. Historical route values, risk route rates, ETA stack features, raw risk probabilities, and calibration inputs are OOF for fitting shipments. Final test maps and models use train+validation only; labels are evaluated after prediction.\n\n## 10. Limitations And Reproduction Scope\n\nThis is a deterministic synthetic reproduction. The final test rerun verifies reproducibility against frozen reference results; it is not a newly blinded, independent, or real-world generalization evaluation. Small route/stage samples and synthetic mechanisms limit operational conclusions.\n"
    (outputs/"FINAL_PIPELINE_REPORT.md").write_text(report,encoding="utf-8")
    cases=[]
    from .src.recommendations import recommendations
    for title,row in (("S1 highest risk",predictions.loc[predictions.snapshot_stage.eq("ORIGIN_DEPARTED")].sort_values(["risk_probability","snapshot_id"],ascending=[False,True]).iloc[0]),("S2 highest structured deviation",predictions.loc[predictions.snapshot_stage.eq("PORT_ARRIVED")].assign(total=lambda x:x.predicted_customs_deviation_hours+x.predicted_post_customs_deviation_hours).sort_values(["total","snapshot_id"],ascending=[False,True]).iloc[0]),("S3 lowest risk",predictions.loc[predictions.snapshot_stage.eq("CUSTOMS_CLEARED")].sort_values(["risk_probability","snapshot_id"]).iloc[0])):
        waterfall = "Not applicable at S1."
        if row.snapshot_stage == "PORT_ARRIVED": waterfall = f"planned customs {row.planned_customs_remaining_hours:.2f}h + customs deviation {row.predicted_customs_deviation_hours:.2f}h + planned post-customs {row.planned_post_customs_remaining_hours:.2f}h + post-customs deviation {row.predicted_post_customs_deviation_hours:.2f}h"
        if row.snapshot_stage == "CUSTOMS_CLEARED": waterfall = f"planned inland {row.planned_inland_remaining_hours:.2f}h + inland deviation {row.predicted_inland_deviation_hours:.2f}h"
        action_text = "\n".join(f"  - {action}" for action in recommendations(row, float(row.risk_probability)))
        cases.append(f"## {title}\n\n- Shipment: `{row.shipment_id}`; route: `{row.route}`; stage: `{row.snapshot_stage}`.\n- Predicted ETA: `{row.predicted_final_eta}`; predicted delay: `{row.predicted_final_delay_hours:.2f}h`.\n- Risk probability: `{row.risk_probability:.3f}`; alert at fixed 0.29: `{bool(row.predicted_material_delay)}`.\n- Waterfall: {waterfall}.\n- Rule-based recommendation:\n{action_text}\n\nActual outcome is shown only after the frozen case-selection rule above: actual final delay `{row.target_final_delay_hours:.2f}h`.\n")
    (outputs/"final_case_studies.md").write_text("# Final Case Studies\n\n"+"\n".join(cases),encoding="utf-8")
    axis=comparison.loc[comparison.scope.eq("ALL")].set_index("method").mae_hours.plot.barh(figsize=(8,4),color="#245b82"); axis.set_xlabel("MAE (hours)"); plt.tight_layout(); plt.savefig(figures/"regression_comparison.png",dpi=140); plt.close()
    axis=predictions.groupby("snapshot_stage").risk_probability.mean().plot.bar(color="#e07a3f",figsize=(6,4)); axis.set_ylabel("Mean calibrated risk probability"); plt.tight_layout(); plt.savefig(figures/"risk_by_stage.png",dpi=140); plt.close()
    axis=snapshots.snapshot_stage.value_counts().reindex(STAGES).plot.bar(color="#3f8c6a",figsize=(6,4)); axis.set_ylabel("Snapshots"); plt.tight_layout(); plt.savefig(figures/"snapshot_counts_by_stage.png",dpi=140); plt.close()
    axis=shipments.groupby("route").final_delay_hours.mean().plot.bar(color="#b94a48",figsize=(7,4)); axis.set_ylabel("Mean final delay (hours)"); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(figures/"mean_delay_by_route.png",dpi=140); plt.close()
    axis=shipments.route.value_counts().plot.bar(color="#245b82",figsize=(7,4)); axis.set_ylabel("Shipments"); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(figures/"shipment_counts_by_route.png",dpi=140); plt.close()
    axis=shipments.carrier.value_counts().plot.bar(color="#245b82",figsize=(7,4)); axis.set_ylabel("Shipments"); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(figures/"shipment_counts_by_carrier.png",dpi=140); plt.close()
    route_duration.plot.bar(stacked=True,figsize=(8,4),color=["#245b82", "#e07a3f", "#3f8c6a"]); plt.ylabel("Planned hours"); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(figures/"planned_duration_by_route.png",dpi=140); plt.close()
    shipments.final_delay_hours.plot.hist(bins=20,figsize=(6,4),color="#e07a3f"); plt.xlabel("Final delay (hours)"); plt.tight_layout(); plt.savefig(figures/"final_delay_distribution.png",dpi=140); plt.close()
    delay_buckets.plot.bar(figsize=(6,4),color="#3f8c6a"); plt.ylabel("Shipments"); plt.tight_layout(); plt.savefig(figures/"final_delay_buckets.png",dpi=140); plt.close()
    route_delay.material_delay_rate.plot.bar(figsize=(7,4),color="#b94a48"); plt.ylabel("Material delay rate"); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(figures/"route_material_delay_rate.png",dpi=140); plt.close()
    route_delay.median_final_delay_hours.plot.bar(figsize=(7,4),color="#e07a3f"); plt.ylabel("Median final delay (hours)"); plt.xticks(rotation=20,ha="right"); plt.tight_layout(); plt.savefig(figures/"route_median_final_delay.png",dpi=140); plt.close()
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5)); axes[0].scatter(shipments.departure_delay_hours, shipments.port_arrival_delay_hours, s=10); axes[0].set(xlabel="Departure delay", ylabel="Port delay"); axes[1].scatter(shipments.port_arrival_delay_hours, shipments.port_arrival_delay_hours + shipments.customs_incremental_delay_hours, s=10); axes[1].set(xlabel="Port delay", ylabel="Customs delay"); axes[2].scatter(shipments.port_arrival_delay_hours + shipments.customs_incremental_delay_hours, shipments.final_delay_hours, s=10); axes[2].set(xlabel="Customs delay", ylabel="Final delay"); plt.tight_layout(); plt.savefig(figures/"delay_propagation_relations.png",dpi=140); plt.close()
    update_summary[["missing_update_rate", "late_update_rate"]].plot.bar(figsize=(8,4)); plt.ylabel("Rate"); plt.tight_layout(); plt.savefig(figures/"missing_late_update_summary.png",dpi=140); plt.close()
    availability.T.plot.bar(figsize=(8,4)); plt.ylabel("Available fraction"); plt.tight_layout(); plt.savefig(figures/"feature_availability_by_stage.png",dpi=140); plt.close()


def run(root: Path, clean: bool = False) -> dict[str, Any]:
    root=Path(root); data=root/"data"; outputs=root/"outputs"; artifacts=outputs/"artifacts"
    if clean:
        # The clean boundary is intentionally limited to generated package data and outputs.
        for directory in (data, outputs):
            if directory.exists():
                for path in sorted(directory.rglob("*"), reverse=True):
                    if path.is_file(): path.unlink()
                    elif path.is_dir(): path.rmdir()
    data.mkdir(parents=True,exist_ok=True); artifacts.mkdir(parents=True,exist_ok=True)
    shipments,events,snapshots=generate_data(); validate(shipments,events,snapshots); manifest=make_manifest(shipments)
    shipments.to_csv(data/"shipments.csv",index=False); events.to_csv(data/"events.csv",index=False); snapshots.to_csv(data/"snapshots.csv",index=False); manifest.to_csv(outputs/"split_manifest.csv",index=False)
    train_ids=set(manifest.loc[manifest.split.eq("train"),"shipment_id"]); validation_ids=set(manifest.loc[manifest.split.eq("validation"),"shipment_id"]); test_ids=set(manifest.loc[manifest.split.eq("test"),"shipment_id"]); trainval_ids=train_ids|validation_ids
    train_rows=snapshots.loc[snapshots.shipment_id.isin(train_ids)].copy(); validation_rows=snapshots.loc[snapshots.shipment_id.isin(validation_ids)].copy(); train_shipments=shipments.loc[shipments.shipment_id.isin(train_ids)].copy()
    # Train-only validation produces all model comparisons without selecting or tuning policy.
    _, validation_eta=fit_eta(train_rows,train_shipments,validation_rows)
    _, validation_predictions=fit_risk_stack(train_rows,train_shipments,validation_rows,validation_eta)
    validation_predictions.attrs["train_shipments"]=train_shipments
    validation_comparison=regression_metrics(validation_predictions)
    validation_comparison.loc[validation_comparison.method.isin(["B0 Scheduled ETA","B1 Route median","B2 Latest observed carry-forward"])].to_csv(outputs/"baseline_metrics_validation.csv",index=False)
    validation_comparison.loc[validation_comparison.method.isin(["Direct HGB v2","Structured HGB v2","Stage-routed v2 policy"])].to_csv(outputs/"eta_validation_metrics.csv",index=False)
    validation_predictions.to_csv(outputs/"eta_validation_predictions.csv",index=False)
    validation_risk = risk_metrics(validation_predictions)
    validation_risk.to_csv(outputs/"risk_validation_metrics.csv",index=False)
    validation_predictions.to_csv(outputs/"risk_validation_predictions.csv",index=False)
    # Only after validation artifacts are complete, refit on train+validation and score test once.
    trainval_rows=snapshots.loc[snapshots.shipment_id.isin(trainval_ids)].copy(); test=snapshots.loc[snapshots.shipment_id.isin(test_ids)].copy(); trainval_shipments=shipments.loc[shipments.shipment_id.isin(trainval_ids)].copy()
    eta_models,eta_test=fit_eta(trainval_rows,trainval_shipments,test)
    risk_models,predictions=fit_risk_stack(trainval_rows,trainval_shipments,test,eta_test)
    predictions.attrs["train_shipments"]=trainval_shipments; comparison=regression_metrics(predictions); risk=risk_metrics(predictions)
    comparison.to_csv(outputs/"final_test_model_comparison.csv",index=False); risk.to_csv(outputs/"final_test_risk_metrics.csv",index=False); predictions.to_csv(outputs/"final_test_predictions.csv",index=False); calibration_table(predictions).to_csv(outputs/"final_test_risk_calibration.csv",index=False)
    predictions[["shipment_id","snapshot_id","snapshot_stage","risk_raw_probability","risk_probability","risk_level","predicted_material_delay","target_is_materially_delayed"]].to_csv(outputs/"final_test_risk_predictions.csv",index=False)
    joblib.dump(eta_models,artifacts/"eta_v2.joblib"); joblib.dump(risk_models["risk_model"],artifacts/"risk_hgb_v2_stack.joblib"); joblib.dump(risk_models["calibrator"],artifacts/"platt_calibrator.joblib")
    summary={"policy":"Stage-routed v2 ETA plus Risk HGB v2 Stack","seed":SEED,"split_shipments":{"train":175,"validation":37,"test":38},"test_snapshots":int(len(test)),"test_snapshots_by_stage":test.snapshot_stage.value_counts().sort_index().to_dict(),"risk_threshold":RISK_THRESHOLD,"no_post_test_tuning":True}
    (outputs/"final_test_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); write_reports(root,shipments,events,snapshots,manifest,comparison,risk,predictions,validation_comparison,validation_risk)
    return summary
