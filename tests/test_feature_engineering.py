import pandas as pd

from src.data_processing import clean_transactions
from src.feature_engineering import build_snapshot


def test_snapshot_uses_only_observation_window_for_features_and_future_for_label():
    rows = [
        ("A", "SKU-A", 2, "2020-01-15", 10.0, 11111),
        ("A", "SKU-B", 1, "2020-01-15", 5.0, 11111),
        ("B", "SKU-C", 3, "2020-03-15", 10.0, 11111),
        ("FUT-1", "SKU-D", 1, "2020-04-10", 20.0, 11111),
        ("D", "SKU-E", 1, "2020-02-01", 100.0, 22222),
        ("E", "SKU-F", 1, "2020-04-10", 30.0, 33333),
    ]
    transactions = pd.DataFrame(
        [
            {
                "InvoiceNo": invoice,
                "StockCode": sku,
                "Quantity": quantity,
                "InvoiceDate": pd.Timestamp(date),
                "UnitPrice": price,
                "CustomerID": customer_id,
            }
            for invoice, sku, quantity, date, price, customer_id in rows
        ]
    )
    cleaned = clean_transactions(transactions)

    snapshot = build_snapshot(cleaned, "2020-04-01", observation_days=90, prediction_days=30)
    by_customer = snapshot.set_index("CustomerID")

    assert list(by_customer.index) == [11111, 22222]
    customer = by_customer.loc[11111]
    assert customer["line_count"] == 3
    assert customer["order_count"] == 2
    assert customer["total_spend"] == 55.0
    assert customer["average_order_value"] == 27.5
    assert customer["product_diversity"] == 3
    assert customer["recency_days"] == 17.0
    assert customer["mean_days_between_orders"] == 60.0
    assert customer["purchased_next_30d"] == 1

    one_order_customer = by_customer.loc[22222]
    assert pd.isna(one_order_customer["mean_days_between_orders"])
    assert one_order_customer["purchased_next_30d"] == 0
    assert 33333 not in by_customer.index
