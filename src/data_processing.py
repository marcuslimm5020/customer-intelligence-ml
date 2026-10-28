"""Load and clean UCI Online Retail transaction data."""

from pathlib import Path
from typing import Union

import pandas as pd


REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
}


def load_transactions(path: Union[str, Path]) -> pd.DataFrame:
    """Read an Excel or CSV transaction file and return cleaned sale lines."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        transactions = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        transactions = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported transaction file type: {path.suffix}")

    missing = REQUIRED_COLUMNS - set(transactions.columns)
    if missing:
        raise ValueError(f"Transaction data is missing required columns: {sorted(missing)}")

    return clean_transactions(transactions)


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's initial cleaning rules to raw transaction rows.

    Exact duplicates, rows without a customer ID, cancellation invoices,
    non-positive quantities and non-positive unit prices are removed.
    """
    cleaned = transactions.drop_duplicates().copy()
    cleaned["InvoiceNo"] = cleaned["InvoiceNo"].astype("string").str.strip()
    cleaned["StockCode"] = cleaned["StockCode"].astype("string").str.strip()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["CustomerID"] = pd.to_numeric(cleaned["CustomerID"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")

    valid = (
        cleaned["CustomerID"].notna()
        & cleaned["InvoiceNo"].notna()
        & cleaned["InvoiceDate"].notna()
        & ~cleaned["InvoiceNo"].str.startswith("C", na=False)
        & cleaned["Quantity"].gt(0)
        & cleaned["UnitPrice"].gt(0)
    )
    cleaned = cleaned.loc[valid].copy()
    cleaned["CustomerID"] = cleaned["CustomerID"].astype("int64")
    cleaned["LineAmount"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    return cleaned.sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)
