from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from config import RISK_THRESHOLD, SEED
from src.data_split import make_manifest
from src.eda import run_eda
from src.evaluation import calibration_table, regression_metrics, risk_metrics
from src.eta_models import fit_eta
from src.generate_data import generate_data
from src.reporting import output_paths, write_reports
from src.risk_model import fit_risk_stack
from src.validate_data import validate


def run(root: Path, clean: bool = False) -> dict[str, Any]:
    root = Path(root)
    data = root / "data"
    outputs = root / "outputs"
    paths = output_paths(root)
    artifacts = paths["artifacts"]
    if clean:
        # The clean boundary is intentionally limited to generated package data and outputs.
        for directory in (data, outputs):
            if directory.exists():
                for path in sorted(directory.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
    data.mkdir(parents=True, exist_ok=True)
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    shipments, events, snapshots = generate_data()
    validate(shipments, events, snapshots)
    manifest = make_manifest(shipments)
    shipments.to_csv(data / "shipments.csv", index=False)
    events.to_csv(data / "events.csv", index=False)
    snapshots.to_csv(data / "snapshots.csv", index=False)
    manifest.to_csv(paths["split"] / "split_manifest.csv", index=False)
    run_eda(root, shipments, events, snapshots)
    train_ids = set(manifest.loc[manifest.split.eq("train"), "shipment_id"])
    validation_ids = set(manifest.loc[manifest.split.eq("validation"), "shipment_id"])
    test_ids = set(manifest.loc[manifest.split.eq("test"), "shipment_id"])
    trainval_ids = train_ids | validation_ids
    train_rows = snapshots.loc[snapshots.shipment_id.isin(train_ids)].copy()
    validation_rows = snapshots.loc[snapshots.shipment_id.isin(validation_ids)].copy()
    train_shipments = shipments.loc[shipments.shipment_id.isin(train_ids)].copy()
    # Train-only validation produces all model comparisons without selecting or tuning policy.
    _, validation_eta = fit_eta(train_rows, train_shipments, validation_rows)
    _, validation_predictions = fit_risk_stack(
        train_rows, train_shipments, validation_rows, validation_eta
    )
    validation_predictions.attrs["train_shipments"] = train_shipments
    validation_comparison = regression_metrics(validation_predictions)
    baseline_dir = paths["validation"] / "baselines"
    eta_dir = paths["validation"] / "eta"
    risk_dir = paths["validation"] / "risk"
    for directory in (baseline_dir, eta_dir, risk_dir):
        directory.mkdir(parents=True, exist_ok=True)
    validation_comparison.loc[
        validation_comparison.method.isin(
            ["B0 Scheduled ETA", "B1 Route median", "B2 Latest observed carry-forward"]
        )
    ].to_csv(baseline_dir / "baseline_metrics_validation.csv", index=False)
    validation_comparison.loc[
        validation_comparison.method.isin(
            ["Direct HGB v2", "Structured HGB v2", "Stage-routed v2 policy"]
        )
    ].to_csv(eta_dir / "eta_validation_metrics.csv", index=False)
    validation_predictions.to_csv(
        eta_dir / "eta_validation_predictions.csv", index=False
    )
    validation_risk = risk_metrics(validation_predictions)
    validation_risk.to_csv(risk_dir / "risk_validation_metrics.csv", index=False)
    validation_predictions.to_csv(
        risk_dir / "risk_validation_predictions.csv", index=False
    )
    # Only after validation artifacts are complete, refit on train+validation and score test once.
    trainval_rows = snapshots.loc[snapshots.shipment_id.isin(trainval_ids)].copy()
    test = snapshots.loc[snapshots.shipment_id.isin(test_ids)].copy()
    trainval_shipments = shipments.loc[shipments.shipment_id.isin(trainval_ids)].copy()
    eta_models, eta_test = fit_eta(trainval_rows, trainval_shipments, test)
    risk_models, predictions = fit_risk_stack(
        trainval_rows, trainval_shipments, test, eta_test
    )
    predictions.attrs["train_shipments"] = trainval_shipments
    comparison = regression_metrics(predictions)
    risk = risk_metrics(predictions)
    comparison.to_csv(
        paths["final_metrics"] / "final_test_model_comparison.csv", index=False
    )
    risk.to_csv(paths["final_metrics"] / "final_test_risk_metrics.csv", index=False)
    predictions.to_csv(
        paths["final_predictions"] / "final_test_predictions.csv", index=False
    )
    calibration_table(predictions).to_csv(
        paths["final_metrics"] / "final_test_risk_calibration.csv", index=False
    )
    predictions[
        [
            "shipment_id",
            "snapshot_id",
            "snapshot_stage",
            "risk_raw_probability",
            "risk_probability",
            "risk_level",
            "predicted_material_delay",
            "target_is_materially_delayed",
        ]
    ].to_csv(
        paths["final_predictions"] / "final_test_risk_predictions.csv", index=False
    )
    joblib.dump(eta_models, artifacts / "eta_v2.joblib")
    joblib.dump(risk_models["risk_model"], artifacts / "risk_hgb_v2_stack.joblib")
    joblib.dump(risk_models["calibrator"], artifacts / "platt_calibrator.joblib")
    eta_summary = comparison.loc[
        (comparison.method == "Stage-routed v2 policy") & comparison.scope.eq("ALL")
    ].iloc[0]
    risk_summary = risk.loc[risk.scope.eq("ALL")].iloc[0]
    summary = {
        "policy": "Stage-routed v2 ETA plus Risk HGB v2 Stack",
        "seed": SEED,
        "split_shipments": {"train": 175, "validation": 37, "test": 38},
        "test_snapshots": int(len(test)),
        "test_snapshots_by_stage": test.snapshot_stage.value_counts()
        .sort_index()
        .to_dict(),
        "risk_threshold": RISK_THRESHOLD,
        "no_post_test_tuning": True,
        "eta_mae_hours": float(eta_summary.mae_hours),
        "eta_rmse_hours": float(eta_summary.rmse_hours),
        "risk_pr_auc": float(risk_summary.pr_auc),
        "risk_brier_score": float(risk_summary.brier_score),
        "risk_f1": float(risk_summary.f1),
        "report_paths": {
            "quality": "01_data_quality/data_quality_report.md",
            "eda": "03_eda/eda_summary.md",
            "final": "05_final_evaluation/reports/FINAL_PIPELINE_REPORT.md",
            "cases": "05_final_evaluation/reports/final_case_studies.md",
        },
    }
    (paths["final_reports"] / "final_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_reports(
        root,
        shipments,
        events,
        snapshots,
        manifest,
        comparison,
        risk,
        predictions,
        validation_comparison,
        validation_risk,
    )
    return summary
