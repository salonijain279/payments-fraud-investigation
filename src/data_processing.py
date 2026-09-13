"""Load, merge, and clean the IEEE-CIS Fraud Detection dataset.

Source: IEEE-CIS Fraud Detection (Kaggle competition, data provided by Vesta
Corporation). Only the labeled `train_transaction.csv` + `train_identity.csv`
files are used -- Kaggle's `test_*.csv` files have no isFraud ground truth
(they exist only for competition leaderboard scoring), so this project
carves its own chronological train/test split out of the labeled data
instead (see src/model.py).

IMPORTANT: many columns in this dataset are intentionally masked by Vesta.
Per the competition's own data description, C1-C14 are "counting" features,
D1-D15 are "timedelta" features, M1-M9 are "match" flags, V1-V339 are
Vesta-engineered features, and most id_* columns are anonymized identity/
device signals -- but the exact real-world meaning of any specific column is
not disclosed. This project never invents specific meanings for those
columns (e.g. never claims "C3 = number of phone numbers on file"). They are
used as opaque numeric/categorical model inputs and referred to generically
(by column name and family) everywhere, including in SHAP explanations and
the dashboard.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/transactions_clean.parquet")

# Columns whose real-world meaning IS documented by the competition and are
# safe to use in investigator-facing narrative text.
INTERPRETABLE_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card4",  # card network: visa / mastercard / amex / discover
    "card6",  # card type: credit / debit
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]

# Masked column families -- used as model inputs, never given invented
# semantic labels in any narrative/explanation text.
MASKED_PREFIXES = ("C", "D", "M", "V", "id_")


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    idn = pd.read_csv(raw_dir / "train_identity.csv")
    df = tx.merge(idn, on="TransactionID", how="left")
    print(
        f"Loaded {len(tx):,} transactions, {len(idn):,} with identity data "
        f"({len(idn) / len(tx):.1%} match rate)"
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset="TransactionID")
    print(f"Dropped {before - len(df):,} duplicate TransactionIDs")

    # TransactionDT is seconds elapsed from an arbitrary reference point, not
    # a real calendar timestamp (per the competition data description) -- but
    # it IS a valid ordering key, which is what we need for a chronological
    # split and for time-of-day / day-of-week style features.
    df["transaction_day"] = (df["TransactionDT"] // (24 * 60 * 60)).astype("int32")
    df["transaction_hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["transaction_dow"] = (df["transaction_day"] % 7).astype("int8")

    df["isFraud"] = df["isFraud"].astype("int8")
    df = _downcast_numeric(df)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return df


def run(raw_dir: Path = RAW_DIR, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    df = load_raw(raw_dir)
    df = clean(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved cleaned data ({df.shape[0]:,} rows x {df.shape[1]} cols) to {out_path}")
    return df


if __name__ == "__main__":
    run()
