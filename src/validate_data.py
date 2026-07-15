from __future__ import annotations

import pandas as pd

from config import N_SHIPMENTS, STAGES


def validate(
    shipments: pd.DataFrame, events: pd.DataFrame, snapshots: pd.DataFrame
) -> None:
    if len(shipments) != N_SHIPMENTS or shipments.shipment_id.nunique() != N_SHIPMENTS:
        raise ValueError("Invalid shipment population")
    shipment_ids = set(shipments.shipment_id)
    if not set(events.shipment_id).issubset(shipment_ids) or not set(
        snapshots.shipment_id
    ).issubset(shipment_ids):
        raise ValueError("Events or snapshots reference an unknown shipment")
    if events.groupby("shipment_id").size().ne(5).any():
        raise ValueError("Every shipment must have five events")
    if not set(snapshots.snapshot_stage).issubset(STAGES):
        raise ValueError("Unsupported snapshot stage")
    if (
        snapshots.snapshot_id.duplicated().any()
        or snapshots.target_final_delay_hours.isna().any()
    ):
        raise ValueError("Invalid snapshots")
    actual_columns = [
        "actual_departure_at",
        "actual_port_arrival_at",
        "actual_customs_clearance_at",
        "actual_inland_dispatch_at",
        "actual_final_delivery_at",
    ]
    actual_times = shipments[actual_columns].apply(pd.to_datetime, utc=True)
    if (actual_times.diff(axis=1).iloc[:, 1:] < pd.Timedelta(0)).any().any():
        raise ValueError("Shipment actual milestone chronology is invalid")
    missing = events["is_update_missing"].astype(bool)
    reported = pd.to_datetime(events["reported_at"], utc=True)
    actual = pd.to_datetime(events["actual_at"], utc=True)
    if (
        reported.loc[missing].notna().any()
        or reported.loc[~missing].isna().any()
        or (reported.loc[~missing] < actual.loc[~missing]).any()
    ):
        raise ValueError("Event reporting-time contract is invalid")
    s1 = snapshots.snapshot_stage.eq("ORIGIN_DEPARTED")
    s2 = snapshots.snapshot_stage.eq("PORT_ARRIVED")
    future_at_s1 = [
        "observed_port_arrival_delay_hours",
        "observed_customs_delay_hours",
        "truck_availability_score",
    ]
    future_at_s2 = ["observed_customs_delay_hours", "truck_availability_score"]
    if (
        snapshots.loc[s1, future_at_s1].notna().any().any()
        or snapshots.loc[s2, future_at_s2].notna().any().any()
    ):
        raise ValueError("Snapshot contains future-stage evidence")
