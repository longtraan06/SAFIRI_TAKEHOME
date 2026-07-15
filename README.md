# Shipment Journey Intelligence

SAFiRi AI Intern take-home: a reproducible prototype that predicts shipment final ETA and material-delay risk from milestone-based shipment snapshots.

This repository contains the **final frozen pipeline** only. The accompanying technical report covers methodology, experiments, findings, and limitations; this README focuses on setup, execution, and repository navigation.

## Quick start

### Requirements

- Python 3.11+
- Git

The pipeline generates deterministic synthetic data locally. No external data download or credentials are needed.

### Setup and run

```bash
git clone https://github.com/longtraan06/SAFIRI_TAKEHOME.git
cd SAFIRI_TAKEHOME

python3 -m venv .venv-safiri
source .venv-safiri/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python run_pipeline.py --clean --run-tests
```

For Windows PowerShell, activate the environment with:

```powershell
.\.venv-safiri\Scripts\Activate.ps1
```

`--clean` removes generated files under `data/` and `outputs/` only, then recreates them.

### Other commands

```bash
# Run the pipeline without cleaning existing generated artifacts
python run_pipeline.py

# Run tests only
python -m unittest discover -s tests -v
```

## What the pipeline does

One execution performs the following steps:

1. Generate 250 synthetic shipment journeys using fixed seed `20260715`.
2. Validate data chronology, reporting rules, snapshot availability, and target completeness.
3. Split shipments by ID into train, validation, and test groups: `175 / 37 / 38`.
4. Create EDA summaries and figures.
5. Evaluate ETA baselines, train the frozen ETA and risk models, and write validation artifacts.
6. Refit the frozen policy on train plus validation and reproduce the final held-out evaluation.
7. Save metrics, predictions, figures, reports, case studies, and serialized model artifacts.

The terminal prints ETA and Risk summary metrics and the paths to the main generated reports.

## Final model policy

| Task | Frozen policy |
| --- | --- |
| ETA at S1 (origin departed) | Direct HGB v2 |
| ETA at S2/S3 (port arrived/customs cleared) | Structured HGB v2 |
| Delay risk | Risk HGB v2 Stack with Platt calibration |
| Material-delay alert threshold | `0.29` |

The stage-routed ETA policy uses one model per snapshot; it is not an ensemble. Risk probability does not modify the ETA prediction.

## Repository layout

```text
SAFIRI_TAKEHOME/
├── config.py            # Frozen seed, split, threshold, and model policy
├── run_pipeline.py      # Command-line entry point
├── requirements.txt     # Python dependencies
├── src/                 # Data, features, ETA, risk, evaluation, reporting, and orchestration
│   └── orchestrator.py  # End-to-end generation, training, evaluation, and reporting
├── tests/               # Data quality, leakage, reproducibility, and output-contract tests
├── data/                # Generated shipments, events, and snapshots
└── outputs/             # Generated artifacts, organized by pipeline stage
```

`src/orchestrator.py` coordinates the final workflow. The remaining modules under `src/` separate data generation, validation, features, models, evaluation, and reporting for straightforward inspection and testing.

## Generated outputs

| Location | Contents |
| --- | --- |
| `outputs/01_data_quality/` | Data-quality report. |
| `outputs/02_split/` | Shipment-level split manifest. |
| `outputs/03_eda/` | EDA summary and figures. |
| `outputs/04_model_validation/` | Baseline, ETA, and risk validation metrics and predictions. |
| `outputs/05_final_evaluation/metrics/` | Final ETA, risk, and calibration metrics. |
| `outputs/05_final_evaluation/predictions/` | Snapshot-level final ETA and risk predictions. |
| `outputs/05_final_evaluation/reports/` | `FINAL_PIPELINE_REPORT.md` and `final_case_studies.md`. |
| `outputs/05_final_evaluation/artifacts/` | Serialized ETA models, Risk model, and calibrator. |

## Reproducibility notes

- Data generation, split assignment, models, OOF features, and calibration use fixed random states.
- Splits are grouped by `shipment_id`; snapshots from one shipment never cross partitions.
- Historical route features and ETA stack features are constructed out-of-fold for fitting rows.
- The generated final test result is a reproducibility check of the frozen synthetic benchmark, not a new blind or independent evaluation.

For model design, metrics, and limitations, please refer to the technical report submitted with this take-home.