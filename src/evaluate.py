"""Evaluate the saved final model and explain its November test predictions."""

from pathlib import Path
import argparse
import os

import joblib
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from src.train import ARTIFACTS, DATA_PATH, FEATURES, TARGET


MODEL_PATH = ARTIFACTS / "purchase_propensity_pipeline.joblib"
CALIBRATION_PATH = ARTIFACTS / "holdout_calibration.csv"
CAMPAIGN_PATH = ARTIFACTS / "holdout_campaign_scenarios.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Customer snapshot CSV")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Saved model pipeline")
    parser.add_argument("--output", type=Path, default=ARTIFACTS, help="Evaluation output directory")
    parser.add_argument("--snapshot-date", default="2011-11-01", help="Holdout snapshot date")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    snapshots = pd.read_csv(args.data, parse_dates=["snapshot_date"])
    test = snapshots[snapshots["snapshot_date"].eq(pd.Timestamp(args.snapshot_date))].copy()
    if test.empty:
        parser.error(f"No snapshots found for {args.snapshot_date} in {args.data}")
    model = joblib.load(args.model)
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    actual = test[TARGET].astype(int)

    average_precision = average_precision_score(actual, probabilities)
    brier = brier_score_loss(actual, probabilities)
    positive_rate = float(actual.mean())
    print(f"{args.snapshot_date} customers: {len(test):,}; observed purchase rate: {positive_rate:.3f}")
    print(f"Average precision: {average_precision:.4f}; Brier score: {brier:.4f}")

    # Quantile calibration table and expected calibration error.
    calibration = pd.DataFrame({"actual": actual.to_numpy(), "probability": probabilities})
    calibration["bin"] = pd.qcut(calibration["probability"], q=10, duplicates="drop")
    calibration_bins = calibration.groupby("bin", observed=True).agg(
        customers=("actual", "size"), mean_probability=("probability", "mean"),
        observed_purchase_rate=("actual", "mean"),
    ).reset_index()
    calibration_bins["bin"] = calibration_bins["bin"].astype(str)
    calibration_bins.to_csv(args.output / CALIBRATION_PATH.name, index=False)
    ece = np.average(
        np.abs(calibration_bins["mean_probability"] - calibration_bins["observed_purchase_rate"]),
        weights=calibration_bins["customers"],
    )
    print(f"Expected calibration error (10 quantile bins): {ece:.4f}")

    # Campaign capacity scenarios: evaluate the top-ranked customers at several budgets.
    ranking = np.argsort(-probabilities)
    scenarios = []
    for share in (0.05, 0.10, 0.20, 0.30, 0.50):
        count = max(1, int(np.ceil(len(test) * share)))
        selected = ranking[:count]
        precision = float(actual.iloc[selected].mean())
        scenarios.append(
            {
                "capacity_share": share,
                "customers_contacted": count,
                "precision": precision,
                "recall": float(actual.iloc[selected].sum() / actual.sum()),
                "lift_over_population": precision / positive_rate,
                "minimum_probability": float(probabilities[selected[-1]]),
            }
        )
    pd.DataFrame(scenarios).to_csv(args.output / CAMPAIGN_PATH.name, index=False)

    # Precision-recall and calibration plots.
    precision, recall, _ = precision_recall_curve(actual, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    classifier_name = type(model.named_steps["classifier"]).__name__
    ax.plot(recall, precision, label=f"{classifier_name} (AP={average_precision:.3f})")
    ax.axhline(positive_rate, color="grey", linestyle="--", label="Population purchase rate")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{args.snapshot_date} holdout: precision-recall")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.output / "holdout_precision_recall.png", dpi=160)
    plt.close(fig)

    prob_true, prob_pred = calibration_curve(actual, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="XGBoost")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed purchase rate", title="November holdout calibration")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.output / "holdout_calibration.png", dpi=160)
    plt.close(fig)

    # Explain model scores on the imputed feature matrix. Include missing-value
    # indicator columns so the explanations align exactly with the fitted model.
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        print(f"Skipping tree-based SHAP: {classifier_name} is not a supported tree estimator.")
        print(f"Saved evaluation outputs under {args.output}.")
        return
    transformed = imputer.transform(test[FEATURES])
    feature_names = imputer.get_feature_names_out(FEATURES)
    transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=test.index)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(args.output / "shap_feature_importance.csv", index=False)

    contribution_frame = pd.DataFrame(
        shap_values,
        columns=[f"shap_{name}" for name in feature_names],
        index=test.index,
    )
    contribution_frame.insert(0, "purchase_probability", probabilities)
    contribution_frame.insert(0, "purchased_next_30d", actual.to_numpy())
    contribution_frame.insert(0, "snapshot_date", test["snapshot_date"].astype(str).to_numpy())
    contribution_frame.insert(0, "CustomerID", test["CustomerID"].to_numpy())
    contribution_frame.to_csv(args.output / "holdout_shap_contributions.csv", index=False)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, transformed_frame, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(args.output / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Top SHAP features:")
    print(importance.head(8).to_string(index=False))
    print(f"Saved evaluation, campaign, calibration and SHAP outputs under {args.output}.")


if __name__ == "__main__":
    main()
