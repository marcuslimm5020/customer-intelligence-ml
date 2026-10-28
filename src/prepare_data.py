"""Prepare monthly customer snapshots from a transaction file."""

import argparse

from pathlib import Path

import pandas as pd

from src.data_processing import load_transactions
from src.feature_engineering import build_snapshots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "online_retail.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "customer_snapshots.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH, help="CSV or Excel transactions")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output snapshot CSV")
    parser.add_argument("--start-date", default="2011-03-01", help="First snapshot cutoff (inclusive)")
    parser.add_argument("--end-date", default="2011-11-01", help="Last snapshot cutoff (inclusive)")
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    if start_date > end_date:
        parser.error("--start-date must be on or before --end-date")
    if start_date.day != 1 or end_date.day != 1:
        parser.error("--start-date and --end-date must be the first day of a month")

    transactions = load_transactions(args.source)
    cutoffs = pd.date_range(start_date, end_date, freq="MS")
    if len(cutoffs) == 0:
        parser.error("The date range contains no month-start snapshot cutoffs; use dates on the first of a month")
    snapshots = build_snapshots(transactions, cutoffs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(args.output, index=False)
    print(f"Clean transactions: {len(transactions):,}")
    print(f"Customer snapshots: {len(snapshots):,}")
    print(f"Saved: {args.output}")
    print("Rows and positive labels by cutoff:")
    print(
        snapshots.groupby("snapshot_date")["purchased_next_30d"]
        .agg(customers="size", purchasers="sum", purchase_rate="mean")
        .to_string()
    )


if __name__ == "__main__":
    main()
