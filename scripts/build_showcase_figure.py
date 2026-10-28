"""Build the README results figure from the saved training/evaluation CSVs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "docs" / "assets" / "interview-results.png"


def main() -> None:
    metrics = pd.read_csv(ARTIFACTS / "model_metrics.csv")
    campaign = pd.read_csv(ARTIFACTS / "november_campaign_scenarios.csv")
    calibration = pd.read_csv(ARTIFACTS / "november_calibration.csv")
    validation = metrics.query("split == 'validation' and period == 'ALL'").copy()
    test = metrics.query("split == 'test' and period == 'ALL'").iloc[0]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#344054",
            "text.color": "#182230",
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "axes.edgecolor": "#D0D5DD",
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.13, wspace=0.14, hspace=0.34)
    fig.suptitle("Customer purchase propensity", x=0.06, y=0.97, ha="left", fontsize=21, fontweight="bold")
    fig.text(
        0.06,
        0.915,
        "90-day customer history → 30-day purchase outcome · UCI Online Retail",
        fontsize=11,
        color="#475467",
    )

    # Validation model comparison.
    ax = axes[0, 0]
    order = ["logistic_regression", "random_forest", "xgboost"]
    labels = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
    validation = validation.set_index("model").loc[order].reset_index()
    colors = ["#98A2B3", "#98A2B3", "#147D75"]
    bars = ax.bar(labels, validation["average_precision"], color=colors, width=0.62)
    ax.set_ylim(0.55, 0.64)
    ax.set_ylabel("Average precision")
    ax.set_title("Model selection · validation", loc="left")
    ax.grid(axis="y", color="#EAECF0", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, validation["average_precision"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    ax.text(0.02, 0.97, "Sep–Oct 2011", transform=ax.transAxes, va="top", color="#667085", fontsize=9)

    # Contact capacity tradeoff on the holdout.
    ax = axes[0, 1]
    capacity = campaign["capacity_share"] * 100
    ax.plot(capacity, campaign["precision"] * 100, marker="o", color="#147D75", linewidth=2.5)
    ax.set_xlabel("Customers contacted (% of eligible population)")
    ax.set_ylabel("Contacted customers who purchased (%)")
    ax.set_title("Purchase rate by contact capacity · holdout", loc="left")
    ax.set_xticks(capacity, [f"{int(v)}%" for v in capacity])
    ax.grid(color="#EAECF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.scatter([20], [float(test["precision_at_top_20pct"]) * 100], color="#147D75", zorder=5)
    ax.annotate("Top 20%: about 75 out of 100\npurchased within 30 days", (20, 74.85), xytext=(27, 86), textcoords="data", fontsize=9, arrowprops={"arrowstyle": "-", "color": "#667085"})

    # Calibration comparison.
    ax = axes[1, 0]
    ax.plot([0, 100], [0, 100], linestyle="--", color="#98A2B3", linewidth=1.5, label="Perfect calibration")
    ax.plot(calibration["mean_probability"] * 100, calibration["observed_purchase_rate"] * 100, marker="o", color="#147D75", linewidth=2, label="November holdout")
    mean_prediction = float(np.average(calibration["mean_probability"], weights=calibration["customers"]))
    observed_rate = float(np.average(calibration["observed_purchase_rate"], weights=calibration["customers"]))
    ax.scatter([mean_prediction * 100], [observed_rate * 100], marker="D", color="#C2413B", zorder=5)
    ax.annotate("Overall: 33.4% predicted\nvs 42.9% observed", (mean_prediction * 100, observed_rate * 100), xytext=(7, 71), textcoords="data", fontsize=9, arrowprops={"arrowstyle": "-", "color": "#667085"})
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Mean model score (%)")
    ax.set_ylabel("Observed purchase rate (%)")
    ax.set_title("Calibration caveat · November 2011", loc="left")
    ax.grid(color="#EAECF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    # The temporal evaluation split.
    ax = axes[1, 1]
    ax.set_title("Evaluation follows time", loc="left")
    groups = [
        ("Train", "Mar–Aug", 6, "#667085"),
        ("Validate", "Sep–Oct", 2, "#147D75"),
        ("Test", "Nov", 1, "#D18A00"),
    ]
    start = 0
    for name, dates, width, color in groups:
        ax.barh([0], [width], left=start, color=color, height=0.42)
        ax.text(start + width / 2, 0.04, name, color="white", ha="center", va="center", fontweight="bold", fontsize=8 if width == 1 else 10)
        ax.text(start + width / 2, -0.42, dates, color="#475467", ha="center", va="top", fontsize=9)
        start += width
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.72, 0.62)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0, 0.55, "Validation selects the model; the final month is held aside.", fontsize=9, color="#475467")

    fig.text(
        0.06,
        0.035,
        "One retailer · one final holdout month · use scores for ranking, not as calibrated probabilities",
        fontsize=9,
        color="#667085",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
