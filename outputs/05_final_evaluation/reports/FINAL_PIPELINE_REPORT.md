# Final Pipeline Report

## 1. Reproduction Command

`python final_pipeline/run_pipeline.py --clean --run-tests`

## 2. Seed, Dataset, And Grouped Split

Synthetic data uses seed `20260715`: 250 shipments, five events per shipment, and milestone snapshots. Shipment groups are split train/validation/test as `175/37/38`.

## 3. Data Quality

The generated dataset passed population, event-count, snapshot-ID, stage, and target-completeness validation. See `data_quality_report.md`.

## 4. EDA Insights

`eda_summary.md` and `figures/` document shipment route/carrier/stage counts, planned durations, final-delay distribution/buckets, route delay rates and medians, propagation correlations, update quality, and stage feature availability.

## 5. Baseline Results

B0 is scheduled ETA, B1 maps a train-fitted route median, and B2 carries forward the latest available observed delay. Train-only validation baseline results are in `baseline_metrics_validation.csv`.

## 6. ETA Architecture

The frozen ETA policy routes S1 to Direct residual HGB v2 with an OOF route-delay prior. S2/S3 use Structured planned-deviation HGB v2 waterfall components. It is stage routing, not a test-selected ensemble.

## 7. Risk Architecture

Risk HGB v2 Stack uses OOF route material-delay rates and OOF stage-routed ETA features. Platt calibration is fitted only on OOF raw probabilities; the alert threshold is fixed at `0.29`.

## 8. Validation And Test/Reproduction Metrics

Train-only validation uses train shipments only and is saved in the validation CSVs.

### Train-only Validation ETA

| Method | Scope | n | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| Direct HGB v2 | ALL | 106 | 5.033 | 6.315 |
| Direct HGB v2 | ORIGIN_DEPARTED | 36 | 5.507 | 7.191 |
| Direct HGB v2 | PORT_ARRIVED | 34 | 5.678 | 6.697 |
| Direct HGB v2 | CUSTOMS_CLEARED | 36 | 3.950 | 4.833 |
| Structured HGB v2 | ALL | 70 | 4.404 | 5.600 |
| Structured HGB v2 | PORT_ARRIVED | 34 | 5.783 | 6.894 |
| Structured HGB v2 | CUSTOMS_CLEARED | 36 | 3.102 | 4.011 |
| Stage-routed v2 policy | ALL | 106 | 4.778 | 6.187 |
| Stage-routed v2 policy | ORIGIN_DEPARTED | 36 | 5.507 | 7.191 |
| Stage-routed v2 policy | PORT_ARRIVED | 34 | 5.783 | 6.894 |
| Stage-routed v2 policy | CUSTOMS_CLEARED | 36 | 3.102 | 4.011 |

### Train-only Validation Risk

| Method | Scope | PR-AUC | Brier | F1 |
| --- | --- | ---: | ---: | ---: |
| Risk HGB v2 Stack | ALL | 0.803 | 0.177 | 0.724 |
| Risk HGB v2 Stack | ORIGIN_DEPARTED | 0.743 | 0.210 | 0.686 |
| Risk HGB v2 Stack | PORT_ARRIVED | 0.720 | 0.214 | 0.629 |
| Risk HGB v2 Stack | CUSTOMS_CLEARED | 0.904 | 0.110 | 0.857 |

The following final test rerun is reproducibility verification of the frozen synthetic benchmark, **not a new blind or independent evaluation**.

### Final ETA

| Method | Scope | n | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| B0 Scheduled ETA | ALL | 109 | 12.029 | 14.116 |
| B0 Scheduled ETA | ORIGIN_DEPARTED | 36 | 12.080 | 14.307 |
| B0 Scheduled ETA | PORT_ARRIVED | 36 | 11.811 | 13.662 |
| B0 Scheduled ETA | CUSTOMS_CLEARED | 37 | 12.191 | 14.362 |
| B1 Route median | ALL | 109 | 6.957 | 9.360 |
| B1 Route median | ORIGIN_DEPARTED | 36 | 7.265 | 9.637 |
| B1 Route median | PORT_ARRIVED | 36 | 6.593 | 8.977 |
| B1 Route median | CUSTOMS_CLEARED | 37 | 7.012 | 9.451 |
| B2 Latest observed carry-forward | ALL | 109 | 7.659 | 9.825 |
| B2 Latest observed carry-forward | ORIGIN_DEPARTED | 36 | 11.022 | 13.124 |
| B2 Latest observed carry-forward | PORT_ARRIVED | 36 | 8.355 | 10.018 |
| B2 Latest observed carry-forward | CUSTOMS_CLEARED | 37 | 3.708 | 4.376 |
| Direct HGB v2 | ALL | 109 | 4.899 | 6.312 |
| Direct HGB v2 | ORIGIN_DEPARTED | 36 | 6.278 | 7.645 |
| Direct HGB v2 | PORT_ARRIVED | 36 | 4.823 | 6.127 |
| Direct HGB v2 | CUSTOMS_CLEARED | 37 | 3.632 | 4.897 |
| Structured HGB v2 | ALL | 73 | 3.863 | 5.032 |
| Structured HGB v2 | PORT_ARRIVED | 36 | 4.893 | 6.184 |
| Structured HGB v2 | CUSTOMS_CLEARED | 37 | 2.860 | 3.571 |
| Stage-routed v2 policy | ALL | 109 | 4.661 | 6.022 |
| Stage-routed v2 policy | ORIGIN_DEPARTED | 36 | 6.278 | 7.645 |
| Stage-routed v2 policy | PORT_ARRIVED | 36 | 4.893 | 6.184 |
| Stage-routed v2 policy | CUSTOMS_CLEARED | 37 | 2.860 | 3.571 |

### Final Risk

| Method | Scope | PR-AUC | Brier | F1 |
| --- | --- | ---: | ---: | ---: |
| Risk HGB v2 Stack | ALL | 0.659 | 0.182 | 0.678 |
| Risk HGB v2 Stack | ORIGIN_DEPARTED | 0.565 | 0.219 | 0.684 |
| Risk HGB v2 Stack | PORT_ARRIVED | 0.625 | 0.194 | 0.615 |
| Risk HGB v2 Stack | CUSTOMS_CLEARED | 0.815 | 0.133 | 0.737 |

## 9. Leakage Safeguards

Splits are shipment-grouped. All validation fits use train shipments only. Historical route values, risk route rates, ETA stack features, raw risk probabilities, and calibration inputs are OOF for fitting shipments. Final test maps and models use train+validation only; labels are evaluated after prediction.

## 10. Limitations And Reproduction Scope

This is a deterministic synthetic reproduction. The final test rerun verifies reproducibility against frozen reference results; it is not a newly blinded, independent, or real-world generalization evaluation. Small route/stage samples and synthetic mechanisms limit operational conclusions.
