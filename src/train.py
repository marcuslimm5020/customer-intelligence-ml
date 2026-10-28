"""Train and compare customer purchase propensity models chronologically."""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
try:
    from xgboost import XGBClassifier

    XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # XGBoost's native library may need a system OpenMP runtime.
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_snapshots.csv"
ARTIFACTS = PROJECT_ROOT / "artifacts"
MLRUNS = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "customer-intelligence-ml"
TARGET = "purchased_next_30d"
FEATURES = [
    "line_count",
    "order_count",
    "total_spend",
    "average_line_value",
    "product_diversity",
    "total_quantity",
    "recency_days",
    "average_order_value",
    "mean_days_between_orders",
]
DEFAULT_TRAIN_CUTOFFS = ("2011-03-01", "2011-08-01")
DEFAULT_VALIDATION_CUTOFFS = ("2011-09-01", "2011-10-01")
DEFAULT_TEST_CUTOFFS = ("2011-11-01", "2011-11-01")


def month_cutoffs(start: str, end: str, split_name: str) -> pd.DatetimeIndex:
    """Build an inclusive range of month-start snapshot dates."""
    start_date, end_date = pd.Timestamp(start), pd.Timestamp(end)
    if start_date.day != 1 or end_date.day != 1 or start_date > end_date:
        raise ValueError(f"{split_name} dates must be ordered first-of-month dates")
    return pd.date_range(start_date, end_date, freq="MS")


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 without loading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_models() -> dict[str, Pipeline]:
    """Return comparable estimators with imputation fitted on training data."""
    models = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.1, solver="liblinear", max_iter=1500, random_state=42
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        n_jobs=1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
    else:
        print("XGBoost native runtime unavailable; using sklearn boosting fallback.")
        models["hist_gradient_boosting"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )
    return models


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Summarize ranking, a reference 0.5 threshold, and probability quality."""
    predictions = probabilities >= 0.5
    top_n = max(1, int(np.ceil(0.20 * len(probabilities))))
    top_indices = np.argsort(probabilities)[-top_n:]
    return {
        "average_precision": average_precision_score(y_true, probabilities),
        "precision_at_0_5": precision_score(y_true, predictions, zero_division=0),
        "recall_at_0_5": recall_score(y_true, predictions, zero_division=0),
        "precision_at_top_20pct": float(y_true.iloc[top_indices].mean()),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def evaluate(model: Pipeline, frame: pd.DataFrame, split: str, model_name: str) -> list[dict]:
    """Evaluate overall and by snapshot month, preserving temporal detail."""
    probabilities = model.predict_proba(frame[FEATURES])[:, 1]
    records = []
    for period, indices in [("ALL", frame.index)] + list(frame.groupby("snapshot_date").groups.items()):
        subset = frame.loc[indices]
        p = probabilities[frame.index.get_indexer(indices)]
        records.append(
            {
                "split": split,
                "model": model_name,
                "period": str(period),
                "customers": len(subset),
                "positive_rate": float(subset[TARGET].mean()),
                **calculate_metrics(subset[TARGET], p),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Customer snapshot CSV")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS, help="Output directory (use a new directory for each candidate)")
    for split_name, default_range in (
        ("train", DEFAULT_TRAIN_CUTOFFS),
        ("validation", DEFAULT_VALIDATION_CUTOFFS),
        ("test", DEFAULT_TEST_CUTOFFS),
    ):
        parser.add_argument(f"--{split_name}-start", default=default_range[0])
        parser.add_argument(f"--{split_name}-end", default=default_range[1])
    args = parser.parse_args()

    train_cutoffs = month_cutoffs(args.train_start, args.train_end, "training")
    validation_cutoffs = month_cutoffs(args.validation_start, args.validation_end, "validation")
    test_cutoffs = month_cutoffs(args.test_start, args.test_end, "test")
    if train_cutoffs[-1] >= validation_cutoffs[0] or validation_cutoffs[-1] >= test_cutoffs[0]:
        parser.error("training, validation, and test date ranges must be strictly chronological and non-overlapping")

    import mlflow
    import mlflow.sklearn

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLRUNS.as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    frame = pd.read_csv(args.data, parse_dates=["snapshot_date"])
    available_dates = set(frame["snapshot_date"].dropna().unique())
    requested_dates = set(train_cutoffs) | set(validation_cutoffs) | set(test_cutoffs)
    missing_dates = sorted(requested_dates - available_dates)
    if missing_dates:
        parser.error("snapshot data is missing requested dates: " + ", ".join(pd.Timestamp(d).strftime("%Y-%m-%d") for d in missing_dates))
    train = frame[frame["snapshot_date"].isin(train_cutoffs)].copy()
    validation = frame[frame["snapshot_date"].isin(validation_cutoffs)].copy()
    test = frame[frame["snapshot_date"].isin(test_cutoffs)].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Training, validation or test split is empty; rebuild snapshots first.")

    args.artifacts.mkdir(parents=True, exist_ok=True)
    validation_records = []
    validation_scores = {}
    estimators = make_models()
    for name, estimator in estimators.items():
        estimator.fit(train[FEATURES], train[TARGET])
        model_records = evaluate(estimator, validation, "validation", name)
        validation_records.extend(model_records)
        validation_scores[name] = next(
            row["average_precision"] for row in model_records if row["period"] == "ALL"
        )
        validation_overall = next(
            row for row in model_records if row["period"] == "ALL"
        )
        with mlflow.start_run(run_name=f"{name}-validation"):
            mlflow.log_params(
                {
                    "model_family": name,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "train_cutoffs": ",".join(train_cutoffs.strftime("%Y-%m-%d")),
                    "validation_cutoffs": ",".join(validation_cutoffs.strftime("%Y-%m-%d")),
                    "features": ",".join(FEATURES),
                }
            )
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in validation_overall.items()
                    if key in {
                        "average_precision",
                        "precision_at_0_5",
                        "recall_at_0_5",
                        "precision_at_top_20pct",
                        "brier_score",
                    }
                }
            )
        print(
            f"{name}: validation average precision="
            f"{validation_scores[name]:.4f}"
        )

    selected_name = max(validation_scores, key=validation_scores.get)
    print(f"Selected on validation average precision: {selected_name}")

    # Refit the selected model using all pre-test labels, then evaluate on holdout.
    development = pd.concat([train, validation], ignore_index=True)
    final_model = clone(estimators[selected_name])
    final_model.fit(development[FEATURES], development[TARGET])
    test_records = evaluate(final_model, test, "test", selected_name)
    test_overall = next(row for row in test_records if row["period"] == "ALL")
    print(
        "Holdout test: "
        f"AP={test_overall['average_precision']:.4f}, "
        f"precision@top20%={test_overall['precision_at_top_20pct']:.4f}, "
        f"Brier={test_overall['brier_score']:.4f}"
    )

    pd.DataFrame(validation_records + test_records).to_csv(
        args.artifacts / "model_metrics.csv", index=False
    )
    predictions = test[["CustomerID", "snapshot_date", TARGET]].copy()
    predictions["purchase_probability"] = final_model.predict_proba(test[FEATURES])[:, 1]
    predictions.to_csv(args.artifacts / "test_predictions.csv", index=False)
    joblib.dump(final_model, args.artifacts / "purchase_propensity_pipeline.joblib")
    (args.artifacts / "selected_model.txt").write_text(
        f"{selected_name}\nValidation average precision: {validation_scores[selected_name]:.6f}\n",
        encoding="utf-8",
    )

    model_path = args.artifacts / "purchase_propensity_pipeline.joblib"
    data_hash = file_sha256(args.data)
    model_hash = file_sha256(model_path)
    dependency_versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "mlflow"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = None
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_path": str(args.data.resolve()),
        "source_data_sha256": data_hash,
        "model_path": str(model_path.resolve()),
        "model_sha256": model_hash,
        "selected_model": selected_name,
        "features": FEATURES,
        "target": TARGET,
        "split_cutoffs": {
            "train": train_cutoffs.strftime("%Y-%m-%d").tolist(),
            "validation": validation_cutoffs.strftime("%Y-%m-%d").tolist(),
            "test": test_cutoffs.strftime("%Y-%m-%d").tolist(),
        },
        "rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "validation_average_precision_by_model": validation_scores,
        "holdout_metrics": {
            key: float(value)
            for key, value in test_overall.items()
            if key in {
                "average_precision",
                "precision_at_0_5",
                "recall_at_0_5",
                "precision_at_top_20pct",
                "brier_score",
            }
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": dependency_versions,
    }
    (args.artifacts / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with mlflow.start_run(run_name=f"{selected_name}-final-test"):
        from mlflow.models import infer_signature

        mlflow.log_params(
            {
                "selected_model": selected_name,
                "development_rows": len(development),
                "test_rows": len(test),
                "test_cutoffs": ",".join(test_cutoffs.strftime("%Y-%m-%d")),
                "selected_on": "validation average precision",
            }
        )
        mlflow.log_metrics(
            {
                f"test_{key}": float(value)
                for key, value in test_overall.items()
                if key in {
                    "average_precision",
                    "precision_at_0_5",
                    "recall_at_0_5",
                    "precision_at_top_20pct",
                    "brier_score",
                }
            }
        )
        # Float input columns also support values represented as missing at inference.
        input_example = development[FEATURES].head(5).astype(float)
        output_example = final_model.predict_proba(input_example)[:, 1]
        signature = infer_signature(input_example, output_example)
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        mlflow.log_artifact(str(args.artifacts / "model_metrics.csv"), artifact_path="evaluation")
        mlflow.log_artifact(
            str(args.artifacts / "test_predictions.csv"), artifact_path="evaluation"
        )


if __name__ == "__main__":
    main()
