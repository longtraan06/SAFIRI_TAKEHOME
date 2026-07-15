# Final Pipeline

Self-contained frozen final shipment ETA and material-delay evaluation. It generates its own deterministic synthetic data and never imports or reads root-repository `src/` or `data/` at runtime.

## Requirements

Install `requirements.txt` into a Python 3.11+ environment.

## Run

```bash
python final_pipeline/run_pipeline.py --clean --run-tests
python -m unittest discover -s final_pipeline/tests -v
```

`--clean` deletes generated contents under `final_pipeline/data/` and `final_pipeline/outputs/` only.

## Package Layout

`config.py` holds frozen policy constants. `src/` exposes generator, validation, grouped split, EDA, baselines, feature engineering, ETA, risk, evaluation, reporting, and recommendation modules. `pipeline.py` remains a backward-compatible orchestration facade.

This folder contains only the final selected path. The root repository remains research history; Direct/Structured HGB v1, Risk HGB v2 Core, Risk v1, and candidate-selection/tuning logic are deliberately excluded.

## Data And Validation

The generator uses seed `20260715` for 250 synthetic shipments, events, and available milestone snapshots. Validation checks the shipment population, event count, snapshot stages, duplicate snapshot IDs, and target completeness.

## Split And Validation Protocol

Shipment groups are deterministically split `175/37/38` into train/validation/test. Validation trains only on train shipments and writes B0/B1/B2, ETA v2, and Risk Stack validation artifacts. It does not choose a model, calibrator, or threshold. Final reproduction refits once on train+validation and evaluates the untouched test set.

## Frozen Policy

- B0 is scheduled ETA, B1 is route-median final delay, and B2 carries the latest available observed delay forward.
- ETA v2 routes S1 (`ORIGIN_DEPARTED`) to Direct residual HGB v2 and S2/S3 to Structured planned-deviation HGB v2.
- Risk HGB v2 Stack uses shipment-grouped OOF route material-delay rates and OOF stage-routed ETA features.
- Platt calibration is fitted on OOF raw probabilities only. The alert threshold is frozen at `0.29`.

## Outputs

`data/` contains generated inputs. `outputs/` contains `data_quality_report.md`, `split_manifest.csv`, `eda_summary.md`, `figures/`, validation CSVs, final test prediction/metric/calibration CSVs, `final_case_studies.md`, `FINAL_PIPELINE_REPORT.md`, and serialized artifacts.

## Tests

Focused named tests cover deterministic generation, data validation, grouped splitting, leakage-safe feature construction, baselines, ETA stage routing, Risk Stack feature slots/threshold, and the complete output/reporting contract.

## Reproducibility And Limitations

All generator, split, HGB, OOF, and calibration random states use the frozen seed. The final test rerun is **reproducibility verification of the frozen synthetic benchmark, not a new blind or independent evaluation**. Results are not evidence of real-world generalization or causal operational recommendations.
