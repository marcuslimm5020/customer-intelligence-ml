"""Create customer-level features and forward purchase labels."""

from collections.abc import Iterable
from typing import Union

import pandas as pd


def build_snapshot(
    transactions: pd.DataFrame,
    cutoff: Union[str, pd.Timestamp],
    observation_days: int = 90,
    prediction_days: int = 30,
) -> pd.DataFrame:
    """Build customer features at ``cutoff`` and a forward purchase label.

    Features use sale lines in [cutoff - observation_days, cutoff). The
    population consists of customers with at least one sale in that window.
    The label is 1 if the customer has a sale in [cutoff, cutoff +
    prediction_days). No post-cutoff transaction is used as a feature.
    """
    cutoff = pd.Timestamp(cutoff)
    obs_start = cutoff - pd.Timedelta(days=observation_days)
    pred_end = cutoff + pd.Timedelta(days=prediction_days)

    history = transactions.loc[
        transactions["InvoiceDate"].ge(obs_start)
        & transactions["InvoiceDate"].lt(cutoff)
    ].copy()
    future = transactions.loc[
        transactions["InvoiceDate"].ge(cutoff)
        & transactions["InvoiceDate"].lt(pred_end)
    ]

    history["OrderDate"] = history["InvoiceDate"].dt.normalize()
    orders = (
        history.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(OrderDate=("InvoiceDate", "min"), OrderSpend=("LineAmount", "sum"))
    )
    customer = history.groupby("CustomerID").agg(
        line_count=("InvoiceNo", "size"),
        order_count=("InvoiceNo", "nunique"),
        total_spend=("LineAmount", "sum"),
        average_line_value=("LineAmount", "mean"),
        product_diversity=("StockCode", "nunique"),
        total_quantity=("Quantity", "sum"),
        last_purchase=("InvoiceDate", "max"),
    )
    customer["recency_days"] = (
        cutoff - customer.pop("last_purchase")
    ).dt.total_seconds() / 86400
    customer["average_order_value"] = customer["total_spend"] / customer["order_count"]

    order_dates = orders.sort_values(["CustomerID", "OrderDate"]).groupby("CustomerID")["OrderDate"]
    mean_gaps = order_dates.apply(
        lambda dates: dates.diff().dt.total_seconds().dropna().mean() / 86400
    )
    customer["mean_days_between_orders"] = mean_gaps

    future_customers = set(future["CustomerID"].unique())
    customer["purchased_next_30d"] = customer.index.isin(future_customers).astype("int8")
    customer.insert(0, "snapshot_date", cutoff)
    return customer.reset_index().sort_values("CustomerID").reset_index(drop=True)


def build_snapshots(
    transactions: pd.DataFrame,
    cutoffs: Iterable[Union[str, pd.Timestamp]],
    observation_days: int = 90,
    prediction_days: int = 30,
) -> pd.DataFrame:
    """Build and concatenate a set of chronological customer snapshots."""
    frames = [
        build_snapshot(transactions, cutoff, observation_days, prediction_days)
        for cutoff in cutoffs
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
