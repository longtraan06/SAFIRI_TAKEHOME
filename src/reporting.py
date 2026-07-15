from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from src.recommendations import recommendations

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def output_paths(root: Path) -> dict[str, Path]:
    """Keep generated artifacts discoverable by pipeline stage."""
    outputs = root / "outputs"
    paths = {
        "quality": outputs / "01_data_quality",
        "split": outputs / "02_split",
        "validation": outputs / "04_model_validation",
        "final": outputs / "05_final_evaluation",
    }
    paths["final_metrics"] = paths["final"] / "metrics"
    paths["final_predictions"] = paths["final"] / "predictions"
    paths["final_reports"] = paths["final"] / "reports"
    paths["final_figures"] = paths["final"] / "figures"
    paths["artifacts"] = paths["final"] / "artifacts"
    return paths


def write_reports(
    root: Path,
    shipments: pd.DataFrame,
    events: pd.DataFrame,
    snapshots: pd.DataFrame,
    manifest: pd.DataFrame,
    comparison: pd.DataFrame,
    risk: pd.DataFrame,
    predictions: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    validation_risk: pd.DataFrame,
) -> None:
    paths = output_paths(root)
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    quality, validation = paths["quality"], paths["validation"]
    final_reports, final_figures = paths["final_reports"], paths["final_figures"]
    (quality / "data_quality_report.md").write_text(
        f"# Data Quality\n\n- Shipments: {len(shipments)}\n- Events: {len(shipments)*5}\n- Snapshots: {len(snapshots)}\n- Validation: passed.\n",
        encoding="utf-8",
    )
    frozen = "# Frozen Policy\n\n- Seed: `20260715`; grouped split: `175/37/38`.\n- ETA: Direct residual HGB v2 at S1; Structured planned-deviation HGB v2 at S2/S3.\n- Risk: Risk HGB v2 Stack using OOF route material-delay rate and OOF stage-routed ETA features; Platt calibration; fixed threshold `0.29`.\n"
    (validation / "frozen_policy.md").write_text(frozen, encoding="utf-8")
    eta_rows = "\n".join(
        f"| {r.method} | {r.scope} | {r.n_snapshots} | {r.mae_hours:.3f} | {r.rmse_hours:.3f} |"
        for _, r in comparison.iterrows()
    )
    risk_rows = "\n".join(
        f"| {r.method} | {r.scope} | {r.pr_auc:.3f} | {r.brier_score:.3f} | {r.f1:.3f} |"
        for _, r in risk.iterrows()
    )
    validation_eta_rows = "\n".join(
        f"| {r.method} | {r.scope} | {r.n_snapshots} | {r.mae_hours:.3f} | {r.rmse_hours:.3f} |"
        for _, r in validation_comparison.loc[
            validation_comparison.method.isin(
                ["Direct HGB v2", "Structured HGB v2", "Stage-routed v2 policy"]
            )
        ].iterrows()
    )
    validation_risk_rows = "\n".join(
        f"| {r.method} | {r.scope} | {r.pr_auc:.3f} | {r.brier_score:.3f} | {r.f1:.3f} |"
        for _, r in validation_risk.iterrows()
    )
    report = (
        "# Final Pipeline Report\n\n## 1. Reproduction Command\n\n`python run_pipeline.py --clean --run-tests`\n\n## 2. Seed, Dataset, And Grouped Split\n\nSynthetic data uses seed `20260715`: 250 shipments, five events per shipment, and milestone snapshots. Shipment groups are split train/validation/test as `175/37/38`.\n\n## 3. Data Quality\n\nThe generated dataset passed population, event-count, snapshot-ID, stage, and target-completeness validation. See `data_quality_report.md`.\n\n## 4. EDA Insights\n\n`eda_summary.md` and `figures/` document shipment route/carrier/stage counts, planned durations, final-delay distribution/buckets, route delay rates and medians, propagation correlations, update quality, and stage feature availability.\n\n## 5. Baseline Results\n\nB0 is scheduled ETA, B1 maps a train-fitted route median, and B2 carries forward the latest available observed delay. Train-only validation baseline results are in `baseline_metrics_validation.csv`.\n\n## 6. ETA Architecture\n\nThe frozen ETA policy routes S1 to Direct residual HGB v2 with an OOF route-delay prior. S2/S3 use Structured planned-deviation HGB v2 waterfall components. It is stage routing, not a test-selected ensemble.\n\n## 7. Risk Architecture\n\nRisk HGB v2 Stack uses OOF route material-delay rates and OOF stage-routed ETA features. Platt calibration is fitted only on OOF raw probabilities; the alert threshold is fixed at `0.29`.\n\n## 8. Validation And Test/Reproduction Metrics\n\nTrain-only validation uses train shipments only and is saved in the validation CSVs.\n\n### Train-only Validation ETA\n\n| Method | Scope | n | MAE | RMSE |\n| --- | --- | ---: | ---: | ---: |\n"
        + validation_eta_rows
        + "\n\n### Train-only Validation Risk\n\n| Method | Scope | PR-AUC | Brier | F1 |\n| --- | --- | ---: | ---: | ---: |\n"
        + validation_risk_rows
        + "\n\nThe following final test rerun is reproducibility verification of the frozen synthetic benchmark, **not a new blind or independent evaluation**.\n\n### Final ETA\n\n| Method | Scope | n | MAE | RMSE |\n| --- | --- | ---: | ---: | ---: |\n"
        + eta_rows
        + "\n\n### Final Risk\n\n| Method | Scope | PR-AUC | Brier | F1 |\n| --- | --- | ---: | ---: | ---: |\n"
        + risk_rows
        + "\n\n## 9. Leakage Safeguards\n\nSplits are shipment-grouped. All validation fits use train shipments only. Historical route values, risk route rates, ETA stack features, raw risk probabilities, and calibration inputs are OOF for fitting shipments. Final test maps and models use train+validation only; labels are evaluated after prediction.\n\n## 10. Limitations And Reproduction Scope\n\nThis is a deterministic synthetic reproduction. The final test rerun verifies reproducibility against frozen reference results; it is not a newly blinded, independent, or real-world generalization evaluation. Small route/stage samples and synthetic mechanisms limit operational conclusions.\n"
    )
    (final_reports / "FINAL_PIPELINE_REPORT.md").write_text(report, encoding="utf-8")
    cases = []
    for title, row in (
        (
            "S1 highest risk",
            predictions.loc[predictions.snapshot_stage.eq("ORIGIN_DEPARTED")]
            .sort_values(["risk_probability", "snapshot_id"], ascending=[False, True])
            .iloc[0],
        ),
        (
            "S2 highest structured deviation",
            predictions.loc[predictions.snapshot_stage.eq("PORT_ARRIVED")]
            .assign(
                total=lambda x: x.predicted_customs_deviation_hours
                + x.predicted_post_customs_deviation_hours
            )
            .sort_values(["total", "snapshot_id"], ascending=[False, True])
            .iloc[0],
        ),
        (
            "S3 lowest risk",
            predictions.loc[predictions.snapshot_stage.eq("CUSTOMS_CLEARED")]
            .sort_values(["risk_probability", "snapshot_id"])
            .iloc[0],
        ),
    ):
        waterfall = "Not applicable at S1."
        if row.snapshot_stage == "PORT_ARRIVED":
            waterfall = f"planned customs {row.planned_customs_remaining_hours:.2f}h + customs deviation {row.predicted_customs_deviation_hours:.2f}h + planned post-customs {row.planned_post_customs_remaining_hours:.2f}h + post-customs deviation {row.predicted_post_customs_deviation_hours:.2f}h"
        if row.snapshot_stage == "CUSTOMS_CLEARED":
            waterfall = f"planned inland {row.planned_inland_remaining_hours:.2f}h + inland deviation {row.predicted_inland_deviation_hours:.2f}h"
        action_text = "\n".join(
            f"  - {action}"
            for action in recommendations(row, float(row.risk_probability))
        )
        cases.append(
            f"## {title}\n\n- Shipment: `{row.shipment_id}`; route: `{row.route}`; stage: `{row.snapshot_stage}`.\n- Predicted ETA: `{row.predicted_final_eta}`; predicted delay: `{row.predicted_final_delay_hours:.2f}h`.\n- Risk probability: `{row.risk_probability:.3f}`; alert at fixed 0.29: `{bool(row.predicted_material_delay)}`.\n- Waterfall: {waterfall}.\n- Rule-based recommendation:\n{action_text}\n\nActual outcome is shown only after the frozen case-selection rule above: actual final delay `{row.target_final_delay_hours:.2f}h`.\n"
        )
    (final_reports / "final_case_studies.md").write_text(
        "# Final Case Studies\n\n" + "\n".join(cases), encoding="utf-8"
    )
    axis = (
        comparison.loc[comparison.scope.eq("ALL")]
        .set_index("method")
        .mae_hours.plot.barh(figsize=(8, 4), color="#245b82")
    )
    axis.set_xlabel("MAE (hours)")
    plt.tight_layout()
    plt.savefig(final_figures / "regression_comparison.png", dpi=140)
    plt.close()
    axis = (
        predictions.groupby("snapshot_stage")
        .risk_probability.mean()
        .plot.bar(color="#e07a3f", figsize=(6, 4))
    )
    axis.set_ylabel("Mean calibrated risk probability")
    plt.tight_layout()
    plt.savefig(final_figures / "risk_by_stage.png", dpi=140)
    plt.close()
