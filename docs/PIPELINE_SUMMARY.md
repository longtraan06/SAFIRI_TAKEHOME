# SAFiRi Final Pipeline Summary

## 1. Purpose And Scope


The pipeline answers two operational questions at each available shipment snapshot:

1. What is the predicted final-delivery ETA and delay relative to the published schedule?
2. What is the calibrated probability that the final delay will exceed 12 hours?

The system models an ocean-import journey:

```text
Origin departure -> Port arrival -> Customs clearance -> Inland dispatch -> Final delivery
```

It is a deterministic synthetic-data prototype. It is not a production integration, an intervention optimizer, or evidence of causal impact.

## 2. How To Run


```bash
python run_pipeline.py --clean --run-tests
```

`--clean` deletes generated content only from this package's `data/` and `outputs/` directories.
The command performs generation, validation, split creation, EDA, train-only validation, final train-plus-validation refit, reproduction evaluation, reporting, and tests. Its final log prints the primary ETA and risk metrics plus the most useful Markdown report paths.

## 3. Frozen Final Policy

The final package does not repeat model selection, threshold selection, or hyperparameter tuning. It reproduces the previously selected policy exactly.

| Decision | Frozen value |
| --- | --- |
| Synthetic-data seed | `20260715` |
| Shipments | 250 |
| Split | 175 train / 37 validation / 38 test shipment groups |
| OOF folds | 5 grouped by `shipment_id` |
| Material-delay target | `final_delay_hours > 12` |
| ETA S1 policy | Direct HGB v2 |
| ETA S2/S3 policy | Structured HGB v2 |
| ETA model combination | Stage routing, not blending or ensembling |
| HGB internal early stopping | Disabled for all final ETA and Risk v2 HGB models |
| Risk model | Risk HGB v2 Stack |
| Risk calibration | Sigmoid / Platt on group-OOF raw probabilities |
| Risk alert threshold | `0.29` |

The test evaluation in this package is a **reproducibility verification of the frozen synthetic benchmark**. It is not a new blind test or a new independent model-selection exercise.

## 4. Project History And Rationale

### 4.1 Phase 1: Problem And Information Boundary

Phase 1 fixed the prediction contract before implementation:

- S1 is immediately after origin departure. It may use schedule, route, carrier, forecast operational signals, documents, and departure status, but not port/customs/inland/final outcomes.
- S2 is immediately after port arrival. It adds observed port delay, refreshed congestion, and document readiness, but not actual customs, inland, or final outcomes.
- S3 is immediately after customs clearance. It adds observed customs delay and truck availability, but not inland-dispatch or final-delivery outcomes.
- The regression target is `final_delay_hours = actual_final_delivery_at - scheduled_final_eta`.
- The risk target is `is_materially_delayed = final_delay_hours > 12`.
- All snapshots for a shipment must stay in exactly one shipment-level split.

These availability rules are the foundation for every later feature, target, test, and leakage control.

### 4.2 Phase 2: Synthetic Journey Data

The generator creates 250 sequential shipment journeys with five events each. It encodes a transparent synthetic delay-propagation process:

```text
route + carrier + weather
  -> departure and port delay
  -> customs delay, moderated by congestion and documents
  -> inland delay, moderated by truck availability
  -> final delivery delay
```

The generator also produces realistic operational imperfections:

- missing event updates;
- delayed event reporting;
- route/carrier variation;
- bounded noise;
- snapshots only when their triggering milestone update is available.

This supports reproducible testing of data quality, feature availability, ETA updates, and risk prediction while keeping the ground-truth mechanism explicit.

### 4.3 Phase 3: Grouped Split, EDA, And Baselines

Phase 3 established a deterministic split by shipment ID rather than snapshot row. This prevents records for the same journey from leaking into separate train, validation, and test partitions.

Three simple ETA baselines remain in the final package because they are essential reference points:

| Baseline | Prediction rule | Why it remains |
| --- | --- | --- |
| B0 Scheduled ETA | Predict zero final delay | Measures value over published plan alone |
| B1 Route median | Add train-fitted route median final delay | Measures route-history value |
| B2 Latest observed delay carry-forward | Use latest stage-available observed delay | Measures value from the most recent execution signal |

Train-only validation baseline results were:

| Method | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| B0 Scheduled ETA | 12.087h | 15.332h |
| B1 Route median | 7.492h | 9.984h |
| B2 Latest observed carry-forward | 8.027h | 10.614h |

The EDA established that port delay is a useful upstream signal, but customs and inland variability still matter. This motivated a model that can both improve ETA and expose downstream components.

### 4.4 Phase 4: ETA Model Evolution

The original HGB v1 models were useful benchmarks but were not retained in the final package. They used row-level internal early stopping, which can split snapshots from one shipment between fitting and internal control data. Direct HGB v1 was the stronger general ETA model, while Structured HGB v1 produced useful S2/S3 components but had weaker overall accuracy.

ETA v2 addressed the group-safety concern and improved the modeling formulation:

| Change | Reason |
| --- | --- |
| `early_stopping=False` | Avoid row-level internal control splits across a shipment group |
| Direct residual target around `route_prior_final_delay` | Separate route-level historical expectation from snapshot-specific residual evidence |
| OOF route prior | Ensure a train shipment does not receive a prior derived from its own final label |
| Planned-duration deviation targets for Structured model | Make S2/S3 waterfall components operationally interpretable |
| Availability-safe interactions and route-typical observed-delay deviation | Add logistics signal without target/future-event leakage |

The v2 validation result determined the final routing policy:

| Final route | Model | Validation rationale |
| --- | --- | --- |
| S1 `ORIGIN_DEPARTED` | Direct HGB v2 | Structured downstream components are not defined before port arrival; Direct v2 S1 MAE was 5.507h |
| S2 `PORT_ARRIVED` | Structured HGB v2 | Gives customs and post-customs waterfall; S2 MAE was 5.783h |
| S3 `CUSTOMS_CLEARED` | Structured HGB v2 | Gives inland waterfall; S3 MAE was 3.102h |

The final policy is routing, not an average of models. Risk probabilities never feed back into ETA predictions.

### 4.5 Risk Model Evolution And Final Choice

The original Risk v1 classifier was retained only as a historical validation comparator, not as final package code. Risk v2 introduced:

- availability-safe v2 operational features;
- `route_material_delay_rate`, calculated from historical material-delay labels;
- OOF mapping for training rows and train-only mapping for validation rows;
- OOF stage-routed ETA delay as a stacking feature;
- `delay_margin_to_material_threshold = predicted_delay - 12`;
- sigmoid/Platt calibration fit only on grouped OOF raw probabilities.

During validation, three classifiers and one heuristic were compared. The final package preserves only the selected Stack model; the comparison is documented here to explain the selection.

| Candidate | Validation PR-AUC | Brier | F1 | Decision |
| --- | ---: | ---: | ---: | --- |
| Risk classifier v1 | 0.791 | 0.197 | 0.660 | Historical benchmark; excluded from final code |
| Risk HGB v2 Core | 0.795 | 0.185 | 0.738 | Improved but not selected |
| Risk HGB v2 Stack | 0.803 | 0.177 | 0.724 | Selected: best ranking and calibration |
| ETA-delay heuristic | N/A | N/A | 0.716 | Deterministic reference only |

The threshold was selected before validation from train OOF calibrated probabilities by maximizing F1, using higher precision as the tie-breaker. The selected value was frozen at `0.29`; it is not reselected during a clean reproduction run.

## 5. Package Architecture

The final package has no runtime dependency on root-repository source modules.

| Module | Responsibility |
| --- | --- |
| `config.py` | Frozen seed, split counts, stages, target threshold, ETA routing, risk threshold |
| `src/generate_data.py` | Deterministic shipment, event, and snapshot generation |
| `src/validate_data.py` | Population, reference, chronology, report-time, snapshot availability, and target checks |
| `src/data_split.py` | Deterministic shipment-group split manifest |
| `src/eda.py` | EDA statistics, `eda_summary.md`, and EDA-only figures |
| `src/baselines.py` | B0/B1/B2 historical/reference ETA logic |
| `src/feature_engineering.py` | Availability-safe features, OOF folds, historical mapping, interactions, preprocessing |
| `src/eta_models.py` | Direct HGB v2, Structured HGB v2, stage routing, and ETA OOF stacking predictions |
| `src/risk_model.py` | Risk Stack features, OOF risk probabilities, Platt calibration, and frozen alert |
| `src/evaluation.py` | Regression metrics, risk metrics, and calibration/reliability table |
| `src/recommendations.py` | Rule-based operational recommendations from snapshot-available signals |
| `src/reporting.py` | Data-quality/frozen-policy/final reports, case studies, and final-evaluation figures |
| `src/orchestrator.py` | Ordered end-to-end pipeline execution and artifact persistence |
| `run_pipeline.py` | Standalone CLI entry point |

No Direct HGB v1, Structured HGB v1, Risk v1, Risk Core, candidate-selection loop, or threshold-tuning loop is shipped in the final package.

## 6. Execution Flow

`src/orchestrator.py` executes the following fixed sequence.

1. If requested, clean only `data/` and `outputs/`.
2. Generate deterministic synthetic data with the frozen seed.
3. Validate data quality and availability boundaries.
4. Create and save the grouped split manifest.
5. Run EDA on the generated dataset and write EDA artifacts.
6. Fit B0/B1/B2 using train shipments and save validation metrics.
7. Fit ETA v2 on train, predict validation, and save validation artifacts.
8. Create group-OOF ETA features, train/calibrate Risk Stack on train, and evaluate validation without tuning.
9. Refit ETA v2 and Risk Stack on train plus validation.
10. Generate final reproduction predictions for test shipments.
11. Attach labels only for evaluation, metrics, calibration table, reports, figures, and case studies.
12. Serialize final model artifacts and print the completion summary.

## 7. Data, Events, And Snapshots

### 7.1 Generated Tables

| Table | Unit | Main use |
| --- | --- | --- |
| `data/shipments.csv` | One journey per shipment | Ground truth, planned/actual milestones, route context |
| `data/events.csv` | One record per milestone | Event reporting and update-quality analysis |
| `data/snapshots.csv` | One as-of prediction record | ETA/Risk feature input and labels |

The clean run generated 250 shipments, 1,250 events, and 718 available snapshots. Snapshot availability differs slightly by stage because missing triggering event updates suppress snapshots.

### 7.2 Current EDA Results

| Finding | Current value / interpretation |
| --- | --- |
| Final delay distribution | Mean 10.382h; median 9.910h; 108 of 250 shipments exceed 12h |
| Highest route median delay | HO_CHI_MINH-SYDNEY: 12.721h |
| Lowest route median delay | SINGAPORE-MELBOURNE: 5.512h |
| Port delay -> final delay correlation | 0.758, a strong upstream ETA signal |
| Port delay -> customs increment correlation | 0.222, positive but incomplete explanation |
| Customs increment -> inland increment correlation | 0.237, positive but incomplete explanation |
| S1 feature availability | Departure delay available; port/customs/truck evidence unavailable |
| S2 feature availability | Port delay available; customs/truck evidence unavailable |
| S3 feature availability | Customs delay and truck availability become available |

Full route/carrier/stage counts, planned durations, delay buckets, update-quality rates, and feature availability are stored in `outputs/03_eda/eda_summary.md`.

## 8. ETA Feature And Target Contract

### 8.1 Base Availability-Safe Features

All HGB pipelines use train-fitted median imputation with missingness indicators for numeric fields, and train-fitted most-frequent imputation plus one-hot encoding for categorical fields.

Categorical features:

```text
route_id, carrier, snapshot_stage
```

Operational numeric features:

```text
planned_remaining_hours
calendar_day_of_week
observed_departure_delay_hours
observed_port_arrival_delay_hours
observed_customs_delay_hours
truck_availability_score
congestion_score
weather_severity
document_readiness_score
event_completeness_score
```

V2 additions:

```text
route_prior_final_delay                 # Direct ETA only
port_delay_x_congestion
port_delay_x_document_gap
customs_delay_x_truck_shortage          # informative only when S3 inputs exist
observed_delay_vs_route_typical
```

Forbidden inputs include shipment/event identifiers, targets, final outcomes, future actual timestamps, future delays, and ground-truth downstream components.

### 8.2 Direct HGB v2: S1 ETA

Direct HGB v2 predicts a residual:

```text
residual_target = final_delay_hours - route_prior_final_delay
predicted_final_delay = route_prior_final_delay + predicted_residual
```

`route_prior_final_delay` is the historical route median final delay. Training snapshots receive a five-fold shipment-group OOF mapping; prediction snapshots receive a mapping fit from all fitting shipments; unknown routes use the fitting overall median.

Frozen Direct configuration:

```text
learning_rate=0.05
max_iter=150
max_leaf_nodes=12
max_depth=3
min_samples_leaf=15
l2_regularization=1.5
early_stopping=False
random_state=20260715
```

### 8.3 Structured HGB v2: S2 And S3 ETA

At S2, the model predicts two deviations from planned duration:

```text
customs_deviation_hours
post_customs_deviation_hours

predicted_final_eta = snapshot_at
                    + planned_customs_remaining_hours
                    + predicted_customs_deviation_hours
                    + planned_post_customs_remaining_hours
                    + predicted_post_customs_deviation_hours
```

At S3, it predicts the remaining inland deviation:

```text
inland_deviation_hours

predicted_final_eta = snapshot_at
                    + planned_inland_remaining_hours
                    + predicted_inland_deviation_hours
```

S2 models exclude S3-only `observed_customs_delay_hours`, `truck_availability_score`, and their interaction. The arithmetic is tested so planned components and deviations reconstruct the persisted final ETA.

Frozen Structured configuration:

```text
learning_rate=0.05
max_iter=140
max_leaf_nodes=10
max_depth=3
min_samples_leaf=15
l2_regularization=2.0
early_stopping=False
random_state=20260715
```

## 9. Risk HGB v2 Stack

### 9.1 Features

Risk Stack uses the availability-safe operational v2 features plus:

```text
route_material_delay_rate
stage_routed_predicted_final_delay_hours
delay_margin_to_material_threshold
```

`route_material_delay_rate` is the historical proportion of shipment labels with final delay above 12 hours. It is computed as follows:

- training rows: OOF by shipment group, so a shipment does not contribute its own label;
- validation rows: map from all outer-train shipments only;
- final reproduction test rows: map from train-plus-validation shipments only;
- unknown route: use the fitting overall material-delay rate.

The two ETA stack columns are created by five-fold group-OOF stage-routed ETA predictions for training rows. A shipment's Risk feature therefore never comes from an ETA model trained on that shipment's labels. Validation and final test ETA stack predictions use only their corresponding fitting partition.

### 9.2 Model, Calibration, And Alert

Risk HGB v2 Stack uses the same shallow HGB configuration as the Structured v2 models with `early_stopping=False` and `max_iter=140`.

Raw group-OOF risk probabilities are passed to logistic regression sigmoid/Platt calibration. The calibrator is fit on OOF probabilities and labels only. The fixed alert rule is:

```text
predicted_material_delay = calibrated_risk_probability >= 0.29
```

The risk output is a predictive ranking and calibration diagnostic. It does not alter ETA, and it is not causal evidence for an intervention.

## 10. Validation Results

### 10.1 ETA Validation

| Method | Scope | n | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| Direct HGB v2 | All | 106 | 5.033h | 6.315h |
| Direct HGB v2 | S1 | 36 | 5.507h | 7.191h |
| Structured HGB v2 | S2/S3 | 70 | 4.404h | 5.600h |
| Structured HGB v2 | S2 | 34 | 5.783h | 6.894h |
| Structured HGB v2 | S3 | 36 | 3.102h | 4.011h |
| Stage-routed v2 | All | 106 | 4.778h | 6.187h |

### 10.2 Risk Validation

| Scope | n | PR-AUC | Brier | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 106 | 0.803 | 0.177 | 0.679 | 0.776 | 0.724 |
| S1 | 36 | 0.743 | 0.210 | 0.667 | 0.706 | 0.686 |
| S2 | 34 | 0.720 | 0.214 | 0.550 | 0.733 | 0.629 |
| S3 | 36 | 0.904 | 0.110 | 0.833 | 0.882 | 0.857 |

Validation artifacts:

```text
outputs/04_model_validation/baselines/baseline_metrics_validation.csv
outputs/04_model_validation/eta/eta_validation_metrics.csv
outputs/04_model_validation/eta/eta_validation_predictions.csv
outputs/04_model_validation/risk/risk_validation_metrics.csv
outputs/04_model_validation/risk/risk_validation_predictions.csv
```

## 11. Final Reproduction Results

### 11.1 ETA

| Method | Scope | n | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| B0 Scheduled ETA | All | 109 | 12.029h | 14.116h |
| B1 Route median | All | 109 | 6.957h | 9.360h |
| B2 Latest observed carry-forward | All | 109 | 7.659h | 9.825h |
| Direct HGB v2 | All | 109 | 4.899h | 6.312h |
| Structured HGB v2 | S2/S3 | 73 | 3.863h | 5.032h |
| Stage-routed v2 policy | All | 109 | 4.661h | 6.022h |
| Stage-routed v2 policy | S1 | 36 | 6.278h | 7.645h |
| Stage-routed v2 policy | S2 | 36 | 4.893h | 6.184h |
| Stage-routed v2 policy | S3 | 37 | 2.860h | 3.571h |

### 11.2 Risk

| Scope | n | PR-AUC | Brier | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 109 | 0.659 | 0.182 | 0.542 | 0.907 | 0.678 |
| S1 | 36 | 0.565 | 0.219 | 0.542 | 0.929 | 0.684 |
| S2 | 36 | 0.625 | 0.194 | 0.480 | 0.857 | 0.615 |
| S3 | 37 | 0.815 | 0.133 | 0.609 | 0.933 | 0.737 |

The final Risk Stack has a high-recall posture at the frozen `0.29` threshold. Its held-out PR-AUC is lower than its validation PR-AUC. This is reported as a small-sample/synthetic-data limitation, not addressed by post-test tuning.

Final reproduction artifacts:

```text
outputs/05_final_evaluation/metrics/final_test_model_comparison.csv
outputs/05_final_evaluation/metrics/final_test_risk_metrics.csv
outputs/05_final_evaluation/metrics/final_test_risk_calibration.csv
outputs/05_final_evaluation/predictions/final_test_predictions.csv
outputs/05_final_evaluation/predictions/final_test_risk_predictions.csv
outputs/05_final_evaluation/reports/FINAL_PIPELINE_REPORT.md
outputs/05_final_evaluation/reports/final_case_studies.md
outputs/05_final_evaluation/artifacts/
```

## 12. Reporting, Case Studies, And Recommendations

Case selection is prediction-only and frozen before actual outcomes are displayed:

| Case | Selection rule |
| --- | --- |
| S1 high-risk case | Highest calibrated Risk Stack probability; tie-break `snapshot_id` |
| S2 waterfall case | Highest predicted customs plus post-customs deviation; tie-break `snapshot_id` |
| S3 low-risk case | Lowest calibrated Risk Stack probability; tie-break `snapshot_id` |

After selection, case studies show predicted ETA, delay, risk probability, fixed alert status, S2/S3 waterfall components where applicable, a rule-based recommendation, and actual final outcome for evaluation context only.

Recommendation rules use only current snapshot signals:

| Available signal | Operational recommendation |
| --- | --- |
| Positive port delay and high congestion | Escalate port/customs follow-up; verify berth, release, and free-day status |
| Low document readiness | Prioritize document completion and pre-clearance review |
| Positive customs delay and low truck availability | Reserve/expedite truck capacity and confirm inland dispatch |
| Incomplete event coverage | Request a verified operational update |
| High risk without a more specific signal | Monitor closely and request carrier/forwarder exception update |

These are transparent operational heuristics. They do not claim that an action causes a different outcome.

## 13. Leakage Controls And Test Coverage

The package includes automated tests for:

- deterministic generation with the fixed seed;
- shipment/event/snapshot quality, chronology, references, reporting time, and snapshot boundaries;
- deterministic `175/37/38` grouped split with disjoint shipment IDs;
- forbidden identifiers, outcomes, and future fields excluded from feature matrices;
- OOF route-prior safety for ETA;
- OOF route material-delay-rate safety for Risk;
- OOF ETA stacking isolation from same-shipment fitted models;
- disabled internal early stopping for ETA and Risk HGB models;
- ETA deviation/final-ETA arithmetic;
- Platt calibration receiving complete OOF raw probabilities;
- frozen risk threshold `0.29`;
- risk probabilities constrained to `[0, 1]`;
- clean-boundary safety, required artifact paths, and report contents.

The current standalone clean run passes all 14 tests.

## 14. Output Layout

```text
outputs/
├── 01_data_quality/
│   └── data_quality_report.md
├── 02_split/
│   └── split_manifest.csv
├── 03_eda/
│   ├── eda_summary.md
│   └── figures/
├── 04_model_validation/
│   ├── baselines/
│   ├── eta/
│   ├── risk/
│   └── frozen_policy.md
└── 05_final_evaluation/
    ├── artifacts/
    ├── figures/
    ├── metrics/
    ├── predictions/
    └── reports/
```

## 15. Limitations And Appropriate Interpretation

- Data is entirely synthetic and depends on the encoded generator assumptions.
- Only four route profiles, three carriers, 250 shipments, and small stage-level validation/test samples are available.
- Missing and delayed updates are simulated and may not match production data bias or operational complexity.
- The final test reproduction verifies implementation consistency against a frozen benchmark; it is not a fresh independent evaluation.
- Regression, risk probabilities, permutation-style feature relationships, and recommendations are predictive aids, not causal effects.
- A production system would require real event ingestion, richer route coverage, customer/SLA-specific material-delay definitions, monitoring, retraining controls, and calibration drift monitoring.
