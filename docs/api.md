# Local prediction API

At startup, the FastAPI application loads the active model recorded in `artifacts/model_registry.json`. If there is no registry entry, it falls back to `artifacts/purchase_propensity_pipeline.joblib`. Set `CUSTOMER_PURCHASE_MODEL` to an explicit model file path to override both. The `/health` response includes the loaded model path. Create a model first with `python -m src.train`, then start the local server from the project root:

```bash
source .venv/bin/activate
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive OpenAPI page. `GET /health` reports whether the model loaded. `POST /predict` accepts customer-level features from the prior 90 days and returns a raw probability score, a thresholded class and the threshold used.

Example request:

```json
{
  "customer_id": "12346",
  "features": {
    "line_count": 20,
    "order_count": 2,
    "total_spend": 680.50,
    "average_line_value": 34.03,
    "product_diversity": 16,
    "total_quantity": 240,
    "recency_days": 12.5,
    "average_order_value": 340.25,
    "mean_days_between_orders": 31.0
  }
}
```

The default class threshold is 0.5. Set `CUSTOMER_PURCHASE_THRESHOLD` before starting Uvicorn to change it. This threshold is a reference choice, not a campaign recommendation. The response labels `purchase_probability` as a raw model score because the November holdout showed under-calibration: the average score was 33.4%, against a 42.9% observed purchase rate. The endpoint is intended for local portfolio demonstration; probability calibration and campaign thresholds should be revisited before operational use.

## Evaluate and promote a candidate

Train into a separate candidate directory, then evaluate using its holdout date and matching snapshot file:

```bash
python -m src.evaluate \
  --data data/processed/customer_snapshots_refresh.csv \
  --model artifacts/candidate_2024_06/purchase_propensity_pipeline.joblib \
  --output artifacts/candidate_2024_06 \
  --snapshot-date 2024-05-01
```

Review `model_metrics.csv` from training and the evaluation outputs before promotion. Promotion changes the registry pointer used at the next API startup:

```bash
python -m src.model_registry --promote artifacts/candidate_2024_06
```

Training writes `model_manifest.json` with the input-data checksum, model checksum, features, split dates, row counts, metrics, and runtime package versions. Promotion checks the model checksum against that manifest; the API checks it again when loading a registered model.

To restore the previous model, run:

```bash
python -m src.model_registry --rollback
```

Restart the API after promotion or rollback; a running process keeps the model it loaded at startup. The local JSON registry is not a shared production model registry or durable backup.
