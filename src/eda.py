from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from config import MATERIAL_DELAY_HOURS, STAGES

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_eda(
    root: Path,
    shipments: pd.DataFrame,
    events: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> None:
    eda_dir = Path(root) / "outputs" / "03_eda"
    figures = eda_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    counts = snapshots.snapshot_stage.value_counts().reindex(STAGES, fill_value=0)
    shipment_counts_route = shipments.route.value_counts()
    shipment_counts_carrier = shipments.carrier.value_counts()
    route_duration = shipments.groupby("route")[
        ["planned_ocean_hours", "planned_customs_hours", "planned_inland_hours"]
    ].median()
    route_delay = (
        shipments.assign(material=shipments.final_delay_hours.gt(MATERIAL_DELAY_HOURS))
        .groupby("route")
        .agg(
            shipments=("shipment_id", "size"),
            material_delay_rate=("material", "mean"),
            median_final_delay_hours=("final_delay_hours", "median"),
        )
    )
    delay_buckets = (
        pd.cut(
            shipments.final_delay_hours,
            [-np.inf, 0, 6, 12, np.inf],
            labels=["early_or_on_time", "1_to_6h", "6_to_12h", "over_12h"],
        )
        .value_counts()
        .reindex(["early_or_on_time", "1_to_6h", "6_to_12h", "over_12h"], fill_value=0)
    )
    propagation = pd.DataFrame(
        {
            "port_delay_to_customs_increment": shipments.port_arrival_delay_hours.corr(
                shipments.customs_incremental_delay_hours
            ),
            "customs_increment_to_inland_increment": shipments.customs_incremental_delay_hours.corr(
                shipments.inland_incremental_delay_hours
            ),
            "port_delay_to_final_delay": shipments.port_arrival_delay_hours.corr(
                shipments.final_delay_hours
            ),
        },
        index=["correlation"],
    ).T
    update_summary = events.groupby("milestone").agg(
        events=("shipment_id", "size"),
        missing_update_rate=("is_update_missing", "mean"),
        late_update_rate=("is_late_update", "mean"),
    )
    availability = (
        snapshots.groupby("snapshot_stage")[
            [
                "observed_departure_delay_hours",
                "observed_port_arrival_delay_hours",
                "observed_customs_delay_hours",
                "truck_availability_score",
            ]
        ]
        .apply(lambda x: x.notna().mean())
        .reindex(STAGES)
    )

    def table(frame: pd.DataFrame) -> str:
        return "\n".join(
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in frame.reset_index().itertuples(index=False, name=None)
        )

    eda_summary = (
        "# EDA Summary\n\n## Shipment Counts By Route\n\n| Route | Shipments |\n| --- | ---: |\n"
        + table(shipment_counts_route.rename("shipments").to_frame())
    )
    eda_summary += (
        "\n\n## Shipment Counts By Carrier\n\n| Carrier | Shipments |\n| --- | ---: |\n"
        + table(shipment_counts_carrier.rename("shipments").to_frame())
    )
    eda_summary += (
        "\n\n## Snapshot Counts By Stage\n\n| Stage | Snapshots |\n| --- | ---: |\n"
        + table(counts.rename("snapshots").to_frame())
    )
    eda_summary += (
        "\n\n## Planned Duration By Route\n\n| Route | Ocean | Customs | Inland |\n| --- | ---: | ---: | ---: |\n"
        + table(route_duration)
    )
    eda_summary += (
        "\n\n## Final Delay Distribution And Buckets\n\n"
        + f"Mean={shipments.final_delay_hours.mean():.3f}h; median={shipments.final_delay_hours.median():.3f}h; standard deviation={shipments.final_delay_hours.std():.3f}h.\n\n| Bucket | Shipments |\n| --- | ---: |\n"
        + table(delay_buckets.rename("shipments").to_frame())
    )
    eda_summary += (
        "\n\n## Route Material Delay Rate And Median\n\n| Route | Shipments | Material delay rate | Median final delay hours |\n| --- | ---: | ---: | ---: |\n"
        + table(route_delay)
    )
    eda_summary += (
        "\n\n## Delay Propagation Relations\n\n| Relation | Correlation |\n| --- | ---: |\n"
        + table(propagation)
    )
    eda_summary += (
        "\n\n## Missing And Late Update Summary\n\n| Milestone | Events | Missing rate | Late rate |\n| --- | ---: | ---: | ---: |\n"
        + table(update_summary)
    )
    eda_summary += (
        "\n\n## Feature Availability By Prediction Stage\n\nValues are fractions available at S1/S2/S3.\n\n| Stage | Departure delay | Port delay | Customs delay | Truck availability |\n| --- | ---: | ---: | ---: | ---: |\n"
        + table(availability)
        + "\n"
    )
    (eda_dir / "eda_summary.md").write_text(eda_summary, encoding="utf-8")

    axis = (
        snapshots.snapshot_stage.value_counts()
        .reindex(STAGES)
        .plot.bar(color="#3f8c6a", figsize=(6, 4))
    )
    axis.set_ylabel("Snapshots")
    plt.tight_layout()
    plt.savefig(figures / "snapshot_counts_by_stage.png", dpi=140)
    plt.close()
    axis = (
        shipments.groupby("route")
        .final_delay_hours.mean()
        .plot.bar(color="#b94a48", figsize=(7, 4))
    )
    axis.set_ylabel("Mean final delay (hours)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "mean_delay_by_route.png", dpi=140)
    plt.close()
    axis = shipments.route.value_counts().plot.bar(color="#245b82", figsize=(7, 4))
    axis.set_ylabel("Shipments")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "shipment_counts_by_route.png", dpi=140)
    plt.close()
    axis = shipments.carrier.value_counts().plot.bar(color="#245b82", figsize=(7, 4))
    axis.set_ylabel("Shipments")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "shipment_counts_by_carrier.png", dpi=140)
    plt.close()
    route_duration.plot.bar(
        stacked=True, figsize=(8, 4), color=["#245b82", "#e07a3f", "#3f8c6a"]
    )
    plt.ylabel("Planned hours")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "planned_duration_by_route.png", dpi=140)
    plt.close()
    shipments.final_delay_hours.plot.hist(bins=20, figsize=(6, 4), color="#e07a3f")
    plt.xlabel("Final delay (hours)")
    plt.tight_layout()
    plt.savefig(figures / "final_delay_distribution.png", dpi=140)
    plt.close()
    delay_buckets.plot.bar(figsize=(6, 4), color="#3f8c6a")
    plt.ylabel("Shipments")
    plt.tight_layout()
    plt.savefig(figures / "final_delay_buckets.png", dpi=140)
    plt.close()
    route_delay.material_delay_rate.plot.bar(figsize=(7, 4), color="#b94a48")
    plt.ylabel("Material delay rate")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "route_material_delay_rate.png", dpi=140)
    plt.close()
    route_delay.median_final_delay_hours.plot.bar(figsize=(7, 4), color="#e07a3f")
    plt.ylabel("Median final delay")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures / "route_median_final_delay.png", dpi=140)
    plt.close()
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].scatter(
        shipments.port_arrival_delay_hours,
        shipments.customs_incremental_delay_hours,
        s=10,
    )
    axes[0].set(xlabel="Port delay", ylabel="Customs increment")
    axes[1].scatter(
        shipments.customs_incremental_delay_hours,
        shipments.inland_incremental_delay_hours,
        s=10,
    )
    axes[1].set(xlabel="Customs increment", ylabel="Inland increment")
    axes[2].scatter(
        shipments.port_arrival_delay_hours, shipments.final_delay_hours, s=10
    )
    axes[2].set(xlabel="Port delay", ylabel="Final delay")
    plt.tight_layout()
    plt.savefig(figures / "delay_propagation_relations.png", dpi=140)
    plt.close()
    update_summary[["missing_update_rate", "late_update_rate"]].plot.bar(figsize=(8, 4))
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig(figures / "missing_late_update_summary.png", dpi=140)
    plt.close()
    availability.T.plot.bar(figsize=(8, 4))
    plt.ylabel("Available fraction")
    plt.tight_layout()
    plt.savefig(figures / "feature_availability_by_stage.png", dpi=140)
    plt.close()
