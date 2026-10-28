# Maintenance runbook

This project currently demonstrates an end-to-end ML workflow on a fixed historical UCI dataset. It does not yet ingest new transactions, schedule jobs, deploy the API, or monitor live predictions. Use this runbook to reproduce the portfolio workflow and to guide the changes needed before operating on refreshed data.

## Reproduce the current portfolio run

From the project root, activate the environment and use the existing local dataset:

```bash
source .venv/bin/activate
python -m pytest -q
python -m src.prepare_data
python -m src.train
python -m src.evaluate
python -m src.monitoring
```

Review the generated snapshot counts and data profile before training. Review validation metrics and the chosen model in MLflow. Use the November holdout as a final report for this historical exercise; it has already been evaluated, so do not tune new model decisions against it.

To inspect experiment runs locally:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

To run the API locally after training:

```bash
uvicorn src.api:app --reload
```

Check `GET /health` and try `POST /predict` using the example in [the API guide](api.md). The API returns a raw score that was under-calibrated on the November holdout; do not interpret it as a reliable real-world probability yet.

## When transaction data is refreshed

1. Keep each source extract immutable and record its source, extraction time, row count, date coverage, and checksum. Do not silently replace the original UCI file.
2. Validate schema, date coverage, missing customer IDs, cancellations, duplicates, invalid quantities/prices, and customer counts. Investigate unexpected changes before feature generation.
3. Build snapshots for dates supported by the new source with `python -m src.prepare_data --source ... --output ... --start-date YYYY-MM-01 --end-date YYYY-MM-01`. The preparation command now accepts a source, output, and inclusive month-start cutoff range. Monitoring accepts `--reference`, `--current`, `--reference-through`, and `--current-date`; its defaults remain historical.
4. Wait until the full 30-day outcome window is present before scoring labels for a snapshot. Exclude incomplete horizons.
5. Run the unit suite, rebuild features, and train a candidate with explicit `--train-start/--train-end`, `--validation-start/--validation-end`, and `--test-start/--test-end` ranges. Put it in a new `--artifacts` directory. Compare validation metrics to the current champion. Keep a new, later time period as an untouched holdout; do not repeatedly reuse an already inspected test period. The script verifies requested snapshots exist and that ranges are ordered and non-overlapping, but it cannot verify that the source extract is complete through each label horizon.
6. Review ranking quality (average precision and campaign-capacity precision/recall), probability calibration (Brier score and calibration curve), and performance by month/customer segment. Investigate drift alerts; PSI is a distribution signal, not proof that model quality changed.
7. Review the candidate's training metrics and holdout evaluation. If it meets agreed business and technical criteria, run `python -m src.model_registry --promote <candidate-directory>`. This records a local active-model pointer and preserves the prior pointer for `python -m src.model_registry --rollback`. Restart the API to load the new model. The candidate must contain training reports, holdout reports, and `model_manifest.json`; promotion does not independently judge whether metric thresholds are acceptable. The manifest records data/model checksums, split dates, features, metrics, and package versions.

## Routine cadence for a real deployment

- **Each data batch:** freshness/schema checks, row-quality checks, feature drift, prediction volume, and API health.
- **After labels mature (30 days):** measure precision/recall, average precision, Brier score, and calibration by snapshot and segment.
- **Retraining:** trigger when labeled performance falls below an agreed baseline, meaningful drift is sustained, or enough new seasonal data has accumulated. Do not retrain automatically from PSI alone.
- **Release:** run CI, stage the versioned model, smoke-check the API, then promote with a rollback path.
- **Housekeeping:** back up the MLflow tracking store and model artifacts. The current `mlruns/` and `artifacts/` are local and Git-ignored, and the training script overwrites the default model artifact.

## Current gaps to close before live operation

- Model evaluation accepts explicit data, model, output directory, and holdout date. The API follows the local active-model registry at startup, and a CLI can promote or roll back. Promotion approval remains a human decision, and the registry is local rather than durable/shared.
- Store model artifacts by immutable version rather than overwriting `artifacts/purchase_propensity_pipeline.joblib`.
- Move MLflow and artifact storage to a durable shared location; local folders are not backups.
- Revisit calibration and the campaign threshold with multiple seasonal validation periods and business contact-capacity/cost assumptions.
- Deploy the API with access control, resource limits, request/error logs, health monitoring, and a rollback process.
- Upgrade from Python 3.9 (end of life) and regenerate the model in the supported runtime. The Docker image has not been built in this environment because Docker is unavailable.
- Add deployment and scheduled pipeline automation only after the refreshed-data contract and promotion criteria are defined.
