"""Compare customer feature distributions with a simple PSI drift check."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.train import ARTIFACTS, DATA_PATH, FEATURES


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """Calculate PSI using reference quantiles plus a separate missing bin."""
    reference_values = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    current_values = pd.to_numeric(current, errors="coerce").to_numpy(dtype=float)
    reference_finite = reference_values[np.isfinite(reference_values)]
    current_finite = current_values[np.isfinite(current_values)]

    if len(reference_finite):
        quantiles = np.quantile(reference_finite, np.linspace(0, 1, bins + 1))
        internal_edges = np.unique(quantiles[1:-1])
    else:
        internal_edges = np.array([], dtype=float)
    edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))

    reference_hist = np.histogram(reference_finite, bins=edges)[0]
    current_hist = np.histogram(current_finite, bins=edges)[0]
    reference_counts = np.append(reference_hist, len(reference_values) - len(reference_finite))
    current_counts = np.append(current_hist, len(current_values) - len(current_finite))

    epsilon = 1e-6
    reference_share = (reference_counts + epsilon) / (reference_counts.sum() + epsilon * len(reference_counts))
    current_share = (current_counts + epsilon) / (current_counts.sum() + epsilon * len(current_counts))
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def compare_features(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Return PSI, missingness rates and a heuristic drift status per feature."""
    results = []
    for feature in FEATURES:
        psi = population_stability_index(reference[feature], current[feature])
        reference_missing = float(reference[feature].isna().mean())
        current_missing = float(current[feature].isna().mean())
        if psi < 0.1:
            status = "low"
        elif psi < 0.2:
            status = "moderate"
        else:
            status = "high"
        results.append(
            {
                "feature": feature,
                "psi": psi,
                "reference_missing_rate": reference_missing,
                "current_missing_rate": current_missing,
                "status": status,
            }
        )
    return pd.DataFrame(results).sort_values("psi", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DATA_PATH)
    parser.add_argument("--current", type=Path, default=DATA_PATH)
    parser.add_argument("--reference-through", default="2011-08-01")
    parser.add_argument("--current-date", default="2011-11-01")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "drift_report.csv")
    args = parser.parse_args()

    reference_data = pd.read_csv(args.reference, parse_dates=["snapshot_date"])
    current_data = pd.read_csv(args.current, parse_dates=["snapshot_date"])
    reference = reference_data.loc[
        reference_data["snapshot_date"].le(pd.Timestamp(args.reference_through))
    ]
    current = current_data.loc[
        current_data["snapshot_date"].eq(pd.Timestamp(args.current_date))
    ]
    if reference.empty or current.empty:
        raise ValueError("Reference or current selection is empty; check snapshot dates and paths.")

    report = compare_features(reference, current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(f"Reference rows: {len(reference):,}; current rows: {len(current):,}")
    print(report.to_string(index=False))
    print(f"Saved drift report: {args.output}")


if __name__ == "__main__":
    main()
