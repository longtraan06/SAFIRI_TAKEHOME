from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from config import MATERIAL_DELAY_HOURS, N_SHIPMENTS, SEED, STAGES

ROUTES = (
    dict(
        route="SHANGHAI-SYDNEY",
        origin="Shanghai, CN",
        destination_port="Port Botany, AU",
        final_destination="Sydney, AU",
        weight=0.29,
        planned_ocean_hours=360.0,
        planned_customs_hours=36.0,
        planned_inland_hours=30.0,
        dispatch_buffer_hours=3.0,
        congestion_tendency=0.58,
        ocean_bounds=(300.0, 450.0),
        customs_bounds=(16.0, 112.0),
        inland_bounds=(12.0, 78.0),
    ),
    dict(
        route="SINGAPORE-MELBOURNE",
        origin="Singapore, SG",
        destination_port="Melbourne, AU",
        final_destination="Melbourne, AU",
        weight=0.22,
        planned_ocean_hours=264.0,
        planned_customs_hours=32.0,
        planned_inland_hours=27.0,
        dispatch_buffer_hours=3.0,
        congestion_tendency=0.43,
        ocean_bounds=(215.0, 350.0),
        customs_bounds=(14.0, 96.0),
        inland_bounds=(10.0, 70.0),
    ),
    dict(
        route="HO_CHI_MINH-SYDNEY",
        origin="Ho Chi Minh City, VN",
        destination_port="Port Botany, AU",
        final_destination="Sydney, AU",
        weight=0.24,
        planned_ocean_hours=312.0,
        planned_customs_hours=40.0,
        planned_inland_hours=32.0,
        dispatch_buffer_hours=3.0,
        congestion_tendency=0.64,
        ocean_bounds=(255.0, 410.0),
        customs_bounds=(18.0, 124.0),
        inland_bounds=(12.0, 84.0),
    ),
    dict(
        route="SHENZHEN-BRISBANE",
        origin="Shenzhen, CN",
        destination_port="Brisbane, AU",
        final_destination="Brisbane, AU",
        weight=0.25,
        planned_ocean_hours=336.0,
        planned_customs_hours=34.0,
        planned_inland_hours=28.0,
        dispatch_buffer_hours=3.0,
        congestion_tendency=0.49,
        ocean_bounds=(275.0, 430.0),
        customs_bounds=(15.0, 104.0),
        inland_bounds=(11.0, 74.0),
    ),
)
CARRIERS = {
    "BlueWave Logistics": (0.36, -1.5),
    "Meridian Cargo": (0.34, 0.5),
    "Pacific Bridge": (0.30, 2.2),
}
CARGO_RISK = {
    "general_merchandise": 0.0,
    "electronics": 1.0,
    "apparel": 0.5,
    "industrial_parts": 2.0,
}


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
        carrier = tuple(CARRIERS)[
            int(rng.choice(3, p=[CARRIERS[x][0] for x in CARRIERS]))
        ]
        cargo = str(rng.choice(tuple(CARGO_RISK), p=(0.40, 0.22, 0.20, 0.18)))
        scheduled_departure = datetime(2026, 1, 5, 6) + timedelta(
            hours=int(rng.integers(0, 180 * 24))
        )
        scheduled_port = scheduled_departure + timedelta(
            hours=route["planned_ocean_hours"]
        )
        scheduled_customs = scheduled_port + timedelta(
            hours=route["planned_customs_hours"]
        )
        scheduled_dispatch = scheduled_customs + timedelta(
            hours=route["dispatch_buffer_hours"]
        )
        scheduled_final = scheduled_port + timedelta(
            hours=route["planned_customs_hours"] + route["planned_inland_hours"]
        )
        weather = float(np.clip(rng.beta(2.0, 5.0), 0, 1))
        congestion = float(
            np.clip(rng.normal(route["congestion_tendency"], 0.13), 0, 1)
        )
        documents = float(np.clip(rng.beta(5.0, 2.2) - 0.05 * CARGO_RISK[cargo], 0, 1))
        truck = float(np.clip(rng.beta(4.3, 2.0) - 0.08 * congestion, 0, 1))
        effect = CARRIERS[carrier][1]
        departure_delay = float(
            np.clip(rng.normal(weather * 4 + effect - 0.8, 1.8), -4, 10)
        )
        ocean_increment = float(
            np.clip(
                rng.normal(
                    weather * 10 + congestion * 2 + effect + 0.2 * departure_delay - 3,
                    3.4,
                ),
                -8,
                32,
            )
        )
        actual_departure = scheduled_departure + timedelta(hours=departure_delay)
        actual_port = actual_departure + timedelta(
            hours=route["planned_ocean_hours"] + ocean_increment
        )
        port_delay = hours(scheduled_port, actual_port)
        customs_increment = float(
            np.clip(
                rng.normal(
                    congestion * 9
                    + (1 - documents) * 12
                    + max(port_delay, 0) * 0.18
                    + CARGO_RISK[cargo]
                    - 5,
                    4.2,
                ),
                -8,
                42,
            )
        )
        actual_customs = actual_port + timedelta(
            hours=route["planned_customs_hours"] + customs_increment
        )
        customs_delay = hours(scheduled_customs, actual_customs)
        dispatch_increment = float(
            np.clip(
                rng.normal(
                    max(customs_delay, 0) * 0.08 + (1 - truck) * 3 + congestion - 1, 1.5
                ),
                -2,
                12,
            )
        )
        final_increment = float(
            np.clip(
                rng.normal(
                    max(customs_delay, 0) * 0.08
                    + (1 - truck) * 8
                    + weather * 2
                    + dispatch_increment * 0.1
                    - 3.5,
                    2.8,
                ),
                -5,
                28,
            )
        )
        actual_dispatch = scheduled_dispatch + timedelta(
            hours=customs_delay + dispatch_increment
        )
        actual_final = actual_dispatch + timedelta(
            hours=route["planned_inland_hours"]
            - route["dispatch_buffer_hours"]
            + final_increment
        )
        final_delay = hours(scheduled_final, actual_final)
        shipment = {
            **route,
            "shipment_id": f"SHP-{number:04d}",
            "carrier": carrier,
            "cargo_type": cargo,
            "weather_severity": round(weather, 3),
            "congestion_score": round(congestion, 3),
            "document_readiness_score": round(documents, 3),
            "truck_availability_score": round(truck, 3),
            "scheduled_departure_at": iso(scheduled_departure),
            "scheduled_port_arrival_at": iso(scheduled_port),
            "scheduled_customs_clearance_at": iso(scheduled_customs),
            "scheduled_inland_dispatch_at": iso(scheduled_dispatch),
            "scheduled_final_eta": iso(scheduled_final),
            "actual_departure_at": iso(actual_departure),
            "actual_port_arrival_at": iso(actual_port),
            "actual_customs_clearance_at": iso(actual_customs),
            "actual_inland_dispatch_at": iso(actual_dispatch),
            "actual_final_delivery_at": iso(actual_final),
            "departure_delay_hours": hours(scheduled_departure, actual_departure),
            "port_arrival_delay_hours": port_delay,
            "customs_incremental_delay_hours": round(customs_increment, 3),
            "inland_incremental_delay_hours": round(
                dispatch_increment + final_increment, 3
            ),
            "final_delay_hours": final_delay,
            "is_materially_delayed": int(final_delay > MATERIAL_DELAY_HOURS),
        }
        shipments.append(shipment)
        event_rows = []
        for milestone, scheduled, actual, place in (
            ("ORIGIN_DEPARTED", scheduled_departure, actual_departure, route["origin"]),
            ("PORT_ARRIVED", scheduled_port, actual_port, route["destination_port"]),
            (
                "CUSTOMS_CLEARED",
                scheduled_customs,
                actual_customs,
                route["destination_port"],
            ),
            (
                "INLAND_DISPATCHED",
                scheduled_dispatch,
                actual_dispatch,
                route["destination_port"],
            ),
            (
                "FINAL_DELIVERED",
                scheduled_final,
                actual_final,
                route["final_destination"],
            ),
        ):
            missing = bool(rng.random() < 0.05)
            delay = (
                None
                if missing
                else float(
                    rng.uniform(4, 30) if rng.random() < 0.14 else rng.uniform(0.05, 3)
                )
            )
            event_rows.append(
                dict(
                    shipment_id=shipment["shipment_id"],
                    milestone=milestone,
                    location=place,
                    scheduled_at=iso(scheduled),
                    actual_at=iso(actual),
                    delay_vs_schedule_hours=hours(scheduled, actual),
                    reported_at=(
                        None if missing else iso(actual + timedelta(hours=delay))
                    ),
                    is_update_missing=missing,
                    is_late_update=False if missing else delay > 4,
                    update_delay_hours=None if missing else round(delay, 3),
                )
            )
        events.extend(event_rows)
        for index, stage in enumerate(STAGES):
            trigger = event_rows[index]
            if trigger["is_update_missing"]:
                continue
            available = [
                event
                for event in event_rows[: index + 1]
                if not event["is_update_missing"]
                and event["reported_at"] <= trigger["reported_at"]
            ]
            present = {event["milestone"] for event in available}
            row = dict(
                snapshot_id=f"{shipment['shipment_id']}-{index+1}",
                shipment_id=shipment["shipment_id"],
                snapshot_stage=stage,
                snapshot_at=trigger["reported_at"],
                origin=route["origin"],
                destination_port=route["destination_port"],
                final_destination=route["final_destination"],
                route=route["route"],
                carrier=carrier,
                cargo_type=cargo,
                planned_ocean_hours=route["planned_ocean_hours"],
                planned_customs_hours=route["planned_customs_hours"],
                planned_inland_hours=route["planned_inland_hours"],
                scheduled_departure_at=iso(scheduled_departure),
                scheduled_port_arrival_at=iso(scheduled_port),
                scheduled_customs_clearance_at=iso(scheduled_customs),
                scheduled_inland_dispatch_at=iso(scheduled_dispatch),
                scheduled_final_eta=iso(scheduled_final),
                weather_severity=round(weather, 3),
                congestion_score=round(congestion, 3),
                document_readiness_score=round(documents, 3),
                observed_departure_delay_hours=np.nan,
                observed_port_arrival_delay_hours=np.nan,
                observed_customs_delay_hours=np.nan,
                truck_availability_score=np.nan,
                upstream_missing_update_count=index + 1 - len(available),
                event_completeness_score=round(len(available) / (index + 1), 3),
                target_final_delay_hours=final_delay,
                target_is_materially_delayed=int(final_delay > MATERIAL_DELAY_HOURS),
                target_actual_final_delivery_at=iso(actual_final),
            )
            if "ORIGIN_DEPARTED" in present:
                row["observed_departure_delay_hours"] = shipment[
                    "departure_delay_hours"
                ]
            if "PORT_ARRIVED" in present:
                row["observed_port_arrival_delay_hours"] = shipment[
                    "port_arrival_delay_hours"
                ]
            if "CUSTOMS_CLEARED" in present:
                row["observed_customs_delay_hours"] = customs_delay
                row["truck_availability_score"] = shipment["truck_availability_score"]
            snapshots.append(row)
    return pd.DataFrame(shipments), pd.DataFrame(events), pd.DataFrame(snapshots)
