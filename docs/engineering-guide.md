# Engineering and monitoring guide

## Experiment tracking with MLflow

`python -m src.train` records each validation candidate and the final holdout run in a local MLflow experiment named `customer-intelligence-ml`. The default tracking store is the ignored `mlruns/` directory. To open the UI:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Set `MLFLOW_TRACKING_URI` before training to use another tracking store. Validation metrics choose the model; the November holdout remains a final evaluation and is not used to select the candidate.

## Feature drift check

Compare the March–August training snapshots with the November snapshot:

```bash
python -m src.monitoring
```

The report uses population stability index (PSI) with reference quantile bins and a separate missing-value bin. The built-in `low`, `moderate`, and `high` status thresholds (<0.1, 0.1–0.2, >=0.2) are heuristics. They flag distribution changes; they do not say whether accuracy has degraded. Override the data or dates for newer snapshots with `--reference`, `--current`, `--reference-through`, and `--current-date`.

For the current historical comparison, all feature PSI values are below 0.1 (the largest is `recency_days` at 0.074). The November purchase rate still rose to 42.9%, which this feature-only PSI check cannot detect; outcome or concept drift needs ongoing labeled data.

## Docker API image

Train the model first so the local `artifacts/purchase_propensity_pipeline.joblib` exists, then build and run:

```bash
docker build -t customer-intelligence-ml-api .
docker run --rm -p 8000:8000 customer-intelligence-ml-api
```

The image installs only the API runtime requirements and Linux OpenMP runtime, runs as a non-root user, and serves the API on port 8000. The model artifact is excluded from source control and supplied from the local build context.

The image currently uses Python 3.9 to match the local serialized model environment. Python 3.9 is past end of life; before treating the image as production software, regenerate the model under a supported Python version and rebuild the image with that matching runtime.

## Continuous integration

GitHub Actions runs the unit suite on pushes, pull requests, and manual dispatch. It installs only API and test dependencies plus the Linux OpenMP runtime; it does not download the retail dataset or require model artifacts.
