import pandas as pd

from src.data_processing import clean_transactions


def test_clean_transactions_keeps_only_unique_identified_positive_sales():
    valid = {
        "InvoiceNo": "10001",
        "StockCode": "SKU-1",
        "Description": "Item",
        "Quantity": 2,
        "InvoiceDate": "2011-01-15 10:00:00",
        "UnitPrice": 5.0,
        "CustomerID": 12345,
        "Country": "United Kingdom",
    }
    rows = [
        valid,
        valid.copy(),
        {**valid, "InvoiceNo": "C10002"},
        {**valid, "InvoiceNo": "10003", "CustomerID": None},
        {**valid, "InvoiceNo": "10004", "Quantity": -1},
        {**valid, "InvoiceNo": "10005", "UnitPrice": 0},
    ]

    cleaned = clean_transactions(pd.DataFrame(rows))

    assert len(cleaned) == 1
    assert cleaned.loc[0, "CustomerID"] == 12345
    assert cleaned.loc[0, "InvoiceNo"] == "10001"
    assert cleaned.loc[0, "LineAmount"] == 10.0
