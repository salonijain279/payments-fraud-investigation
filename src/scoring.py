"""Produce the investigator-facing risk scoring table.

Scores every transaction in the (chronological) test split, assigns a risk
tier, and -- for anything at or above the Medium tier, i.e. what would
actually enter an alert queue -- generates plain-English SHAP reason codes.

Reason-code labels are written by hand per interpretable feature. Masked
C/D/M/V/id_ columns get a generic, family-level label (e.g. "masked
Vesta-engineered feature V70") and never an invented specific meaning --
consistent with src/features.py and the competition's own data description.
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from model import DATA_PATH, MODEL_PATH, chronological_split, risk_tier

OUTPUTS_DIR = Path("outputs")
SCORED_PATH = OUTPUTS_DIR / "scored_transactions.csv"
REASON_TIER_THRESHOLD = 0.3  # Medium and above get SHAP reason codes
TOP_N_REASONS = 3

FEATURE_LABELS = {
    "TransactionAmt": "transaction amount",
    "log_amount": "transaction amount (log-scaled)",
    "transaction_hour": "hour of day",
    "transaction_dow": "day of week",
    "is_night": "nighttime transaction (12am-6am)",
    "is_weekend": "weekend transaction",
    "ProductCD": "product code",
    "card4": "card network",
    "card6": "card type (credit/debit)",
    "addr1": "billing address code",
    "addr2": "billing address code (secondary)",
    "dist1": "distance feature",
    "dist2": "distance feature (secondary)",
    "has_dist1": "distance feature present",
    "has_dist2": "distance feature present (secondary)",
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "recipient email domain",
    "purchaser_email_is_free": "purchaser uses a free/consumer email domain",
    "has_recipient_email": "recipient email domain present",
    "email_domain_match": "purchaser/recipient email domains match",
    "DeviceType": "device type",
    "DeviceInfo": "device/browser string",
    "has_identity_data": "identity/device data present",
    "is_mobile_device": "mobile device",
    "card_prior_txn_count": "this card's prior transaction count",
    "card_avg_amount_prior": "this card's average past transaction amount",
    "card_max_amount_prior": "this card's largest past transaction amount",
    "seconds_since_card_last_txn": "time since this card's last transaction",
    "amount_vs_card_avg_ratio": "amount vs. this card's typical amount",
    "card_is_first_seen": "first transaction seen for this card",
    "v_features_missing_count": "amount of Vesta-engineered data available",
    "v_features_missing_pct": "amount of Vesta-engineered data available",
    "id_features_missing_count": "amount of identity data available",
}

MASKED_FAMILY_LABELS = {
    "C": "masked counting feature",
    "D": "masked time-delta feature",
    "M": "masked match flag",
    "V": "masked Vesta-engineered feature",
    "id_": "masked identity/device signal",
}


def label_for_feature(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for prefix, label in MASKED_FAMILY_LABELS.items():
        if name.startswith(prefix) and (name[len(prefix):].isdigit() if prefix != "id_" else True):
            return f"{label} ({name})"
    return name


def _format_value(feature: str, value) -> str:
    if pd.isna(value):
        return "missing"
    if feature in ("TransactionAmt",) or feature.endswith("_prior") or "amount" in feature.lower():
        try:
            return f"${float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def reason_text(feature: str, shap_value: float, raw_value=None) -> str:
    label = label_for_feature(feature)
    direction = "increased" if shap_value > 0 else "decreased"
    is_masked = feature not in FEATURE_LABELS
    if raw_value is not None and not is_masked:
        return f"{label} ({_format_value(feature, raw_value)}) {direction} risk"
    return f"{label} {direction} risk"


def build_reason_codes(shap_values, feature_names: list, raw_df: pd.DataFrame, top_n: int = TOP_N_REASONS) -> list:
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    reasons = []
    for row_i, row in enumerate(values):
        idx = np.argsort(-np.abs(row))[:top_n]
        raw_row = raw_df.iloc[row_i]
        reasons.append([
            reason_text(feature_names[i], row[i], raw_row[feature_names[i]]) for i in idx
        ])
    return reasons


def main():
    bundle = joblib.load(MODEL_PATH)
    xgb_model = bundle["xgb_model"]
    feature_cols = bundle["feature_cols"]

    df = pd.read_parquet(DATA_PATH)
    _, test = chronological_split(df)

    X_test = test[feature_cols]
    fraud_probability = xgb_model.predict_proba(X_test)[:, 1]

    scored = test[[
        "TransactionID", "TransactionDT", "transaction_day", "isFraud",
        "TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain",
        "DeviceType",
    ]].copy()
    scored["fraud_probability"] = fraud_probability
    scored["risk_tier"] = scored["fraud_probability"].apply(risk_tier)

    scored["top_reason_1"] = ""
    scored["top_reason_2"] = ""
    scored["top_reason_3"] = ""

    alert_mask = scored["fraud_probability"] >= REASON_TIER_THRESHOLD
    n_alerts = int(alert_mask.sum())
    print(f"Generating SHAP reason codes for {n_alerts:,} alert-tier transactions "
          f"(fraud_probability >= {REASON_TIER_THRESHOLD})...")

    if n_alerts > 0:
        explainer = shap.TreeExplainer(xgb_model)
        X_alerts = X_test.loc[alert_mask]
        shap_values = explainer(X_alerts)
        reasons = build_reason_codes(shap_values, feature_cols, X_alerts)
        reason_cols = pd.DataFrame(
            reasons, index=X_alerts.index,
            columns=["top_reason_1", "top_reason_2", "top_reason_3"],
        )
        scored.loc[alert_mask, ["top_reason_1", "top_reason_2", "top_reason_3"]] = reason_cols

    scored = scored.sort_values("fraud_probability", ascending=False).reset_index(drop=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    scored.to_csv(SCORED_PATH, index=False)
    print(f"Saved {len(scored):,} scored transactions -> {SCORED_PATH}")
    print(scored["risk_tier"].value_counts())

    return scored


if __name__ == "__main__":
    main()
