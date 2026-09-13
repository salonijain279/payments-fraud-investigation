"""Fraud Risk & Investigation Console -- Streamlit dashboard.

Four pages: Fraud Overview, Alert Queue, Transaction Investigation, and a
Threshold Simulator. Built on top of the artifacts produced by
src/model.py and src/scoring.py (run those first, or `make pipeline`).

Data note: IEEE-CIS Fraud Detection is real, anonymized e-commerce
transaction data (not synthetic). TransactionDT is a relative time offset
from an arbitrary reference point, not a calendar timestamp. Many columns
(C*, D*, M*, V*, id_*) are intentionally masked by the data provider (Vesta);
this dashboard never claims to know their specific real-world meaning --
they're labeled generically by family everywhere they appear.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import get_model_feature_columns  # noqa: E402
from model import (  # noqa: E402
    DAILY_REVIEW_CAPACITY,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    chronological_split,
)
from scoring import build_reason_codes  # noqa: E402

OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "xgboost_fraud.pkl"
FEATURES_PATH = ROOT / "data" / "processed" / "transactions_features.parquet"

st.set_page_config(page_title="Fraud Risk & Investigation Console", layout="wide")


# ---------------------------------------------------------------- data I/O --
@st.cache_data
def load_scored() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "scored_transactions.csv", low_memory=False)


@st.cache_data
def load_summary() -> dict:
    with open(OUTPUTS_DIR / "model_summary.json") as f:
        return json.load(f)


@st.cache_data
def load_threshold_sweep() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "threshold_sweep.csv")


@st.cache_data
def load_cost_curve() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "cost_sensitive_threshold.csv")


@st.cache_data
def load_daily_capacity() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "daily_capacity_simulation.csv")


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS_DIR / "feature_importance.csv")


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_test_features():
    df = pd.read_parquet(FEATURES_PATH)
    _, test = chronological_split(df)
    return test.set_index("TransactionID", drop=False)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


# --------------------------------------------------------------- constants --
RISK_COLORS = {
    "Critical": "#c0392b",
    "High": "#e67e22",
    "Medium": "#f1c40f",
    "Low": "#7f8c8d",
}


def risk_tier_from_prob(p: float) -> str:
    if p >= 0.8:
        return "Critical"
    elif p >= 0.6:
        return "High"
    elif p >= 0.3:
        return "Medium"
    return "Low"


def cost_at_threshold(scored: pd.DataFrame, threshold: float) -> dict:
    alerts = scored[scored["fraud_probability"] >= threshold]
    missed = scored[(scored["fraud_probability"] < threshold) & (scored["isFraud"] == 1)]
    tp = int((alerts["isFraud"] == 1).sum())
    fp = int((alerts["isFraud"] == 0).sum())
    fn = int(len(missed))
    precision = tp / len(alerts) if len(alerts) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    cost = fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST
    return {
        "alerts": len(alerts), "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "precision": precision, "recall": recall,
        "estimated_cost": cost,
    }


# ------------------------------------------------------------------- pages --
def page_overview(scored: pd.DataFrame, summary: dict):
    st.title("Fraud Overview")
    st.caption(
        "IEEE-CIS Fraud Detection -- real, anonymized e-commerce transactions. "
        "Metrics below are computed on the chronological test split (the most "
        "recent ~20% of transactions by time), never seen during training."
    )

    total = len(scored)
    flagged = int((scored["risk_tier"] != "Low").sum())
    fraud_rate = scored["isFraud"].mean()
    fraud_value = scored.loc[scored["isFraud"] == 1, "TransactionAmt"].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Flagged (Medium+)", f"{flagged:,}")
    c3.metric("Fraud Rate", f"{fraud_rate:.2%}")
    c4.metric("Fraud Value", f"${fraud_value:,.0f}")
    c5.metric("XGBoost ROC-AUC", f"{summary['xgboost']['roc_auc']:.3f}")
    c6.metric("XGBoost PR-AUC", f"{summary['xgboost']['pr_auc']:.3f}")

    col1, col2 = st.columns(2)

    with col1:
        by_product = (
            scored.groupby("ProductCD")
            .agg(transactions=("TransactionID", "count"), fraud_rate=("isFraud", "mean"))
            .reset_index()
            .sort_values("fraud_rate", ascending=False)
        )
        fig = px.bar(
            by_product, x="ProductCD", y="fraud_rate", text_auto=".2%",
            title="Fraud Rate by Product Code",
        )
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tier_counts = scored["risk_tier"].value_counts().reindex(
            ["Critical", "High", "Medium", "Low"]
        ).reset_index()
        tier_counts.columns = ["risk_tier", "count"]
        fig = px.bar(
            tier_counts, x="risk_tier", y="count", color="risk_tier",
            color_discrete_map=RISK_COLORS, title="Risk Tier Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        trend = (
            scored.groupby("transaction_day")
            .agg(transactions=("TransactionID", "count"), fraud=("isFraud", "sum"))
            .reset_index()
        )
        fig = px.line(trend, x="transaction_day", y="fraud", markers=True,
                      title="Fraud Volume Over Time (test period)")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        amt = scored.copy()
        amt["label"] = amt["isFraud"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.histogram(
            amt, x="TransactionAmt", color="label", log_y=True, nbins=60,
            barmode="overlay", opacity=0.6,
            title="Transaction Amount Distribution (log scale, fraud vs. legit)",
        )
        st.plotly_chart(fig, use_container_width=True)


def page_alert_queue(scored: pd.DataFrame):
    st.title("Alert Queue")
    st.caption("Transactions prioritized by predicted fraud risk. Filter and sort like an investigator triage queue.")

    with st.sidebar:
        st.header("Filters")
        tiers = st.multiselect(
            "Risk tier", ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium"],
        )
        products = st.multiselect(
            "Product code", sorted(scored["ProductCD"].dropna().unique().tolist()),
            default=sorted(scored["ProductCD"].dropna().unique().tolist()),
        )
        min_score, max_score = st.slider("Risk score range", 0.0, 1.0, (0.3, 1.0), 0.01)
        min_amt = float(scored["TransactionAmt"].min())
        max_amt = float(scored["TransactionAmt"].max())
        amt_range = st.slider("Amount range ($)", min_amt, max_amt, (min_amt, max_amt))

    filtered = scored[
        scored["risk_tier"].isin(tiers)
        & scored["ProductCD"].isin(products)
        & scored["fraud_probability"].between(min_score, max_score)
        & scored["TransactionAmt"].between(*amt_range)
    ]

    st.write(f"**{len(filtered):,}** transactions match the current filters "
             f"(of {len(scored):,} total in the test period).")

    display_cols = [
        "TransactionID", "fraud_probability", "risk_tier", "TransactionAmt",
        "ProductCD", "card4", "card6", "P_emaildomain", "DeviceType", "top_reason_1",
    ]
    st.dataframe(
        filtered[display_cols]
        .sort_values("fraud_probability", ascending=False)
        .rename(columns={
            "fraud_probability": "Risk Score", "risk_tier": "Tier",
            "TransactionAmt": "Amount", "top_reason_1": "Top Reason",
        }),
        use_container_width=True,
        height=600,
    )


def page_investigation(scored: pd.DataFrame, bundle: dict):
    st.title("Transaction Investigation")

    alert_ids = scored.loc[scored["risk_tier"] != "Low", "TransactionID"].tolist()
    default_id = alert_ids[0] if alert_ids else scored["TransactionID"].iloc[0]
    txn_id = st.selectbox(
        "Select a Transaction ID to investigate",
        options=sorted(scored["TransactionID"].tolist()),
        index=sorted(scored["TransactionID"].tolist()).index(default_id),
    )

    row = scored.loc[scored["TransactionID"] == txn_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Probability", f"{row['fraud_probability']:.3f}")
    col2.metric("Risk Tier", row["risk_tier"])
    col3.metric("Amount", f"${row['TransactionAmt']:,.2f}")

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Transaction Details")
        st.table(pd.DataFrame({
            "Field": ["Transaction ID", "Product Code", "Card Network", "Card Type",
                      "Purchaser Email Domain", "Device Type"],
            "Value": [row["TransactionID"], row["ProductCD"], row["card4"], row["card6"],
                      row["P_emaildomain"], row["DeviceType"]],
        }).set_index("Field"))
        if not pd.isna(row.get("isFraud")):
            st.caption(
                f"Demo-only ground truth (would NOT be known to an investigator "
                f"at review time): isFraud = {int(row['isFraud'])}"
            )

    with d2:
        st.subheader("Why Flagged")
        test_features = load_test_features()
        model = bundle["xgb_model"]
        feature_cols = bundle["feature_cols"]

        if txn_id in test_features.index:
            x_row = test_features.loc[[txn_id], feature_cols]
            explainer = get_explainer(model)
            sv = explainer(x_row)
            reasons = build_reason_codes(sv, feature_cols, x_row, top_n=8)[0]

            contrib = pd.DataFrame({
                "feature": feature_cols,
                "shap_value": sv.values[0],
            }).assign(abs_shap=lambda d: d["shap_value"].abs()).nlargest(10, "abs_shap")
            contrib = contrib.sort_values("shap_value")

            fig = go.Figure(go.Bar(
                x=contrib["shap_value"], y=contrib["feature"], orientation="h",
                marker_color=["#c0392b" if v > 0 else "#2980b9" for v in contrib["shap_value"]],
            ))
            fig.update_layout(
                title="Top SHAP Contributions (red = increases risk, blue = decreases)",
                xaxis_title="SHAP value", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Reason codes:**")
            for r in reasons[:5]:
                st.markdown(f"- {r}")
        else:
            st.info("This transaction is outside the cached test-set feature table.")

    st.divider()
    st.caption(
        "This score indicates **elevated risk that warrants review** -- it is "
        "not a determination that the transaction is fraudulent."
    )


def page_threshold_simulator(scored: pd.DataFrame, summary: dict, sweep: pd.DataFrame, cost_curve: pd.DataFrame):
    st.title("Threshold Simulator")
    st.caption(
        "Move the slider to see how the review threshold trades off alert "
        "volume, fraud capture, and estimated cost. Costs are scenario "
        f"assumptions (false negative = \\${FALSE_NEGATIVE_COST}, false positive "
        f"review = \\${FALSE_POSITIVE_COST}), not real industry figures."
    )

    default_t = float(summary.get("cost_optimal_threshold", 0.5))
    threshold = st.slider("Risk threshold", 0.0, 1.0, default_t, 0.01)

    m = cost_at_threshold(scored, threshold)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Alerts Generated", f"{m['alerts']:,}")
    c2.metric("Frauds Captured", f"{m['true_positives']:,}")
    c3.metric("Precision", f"{m['precision']:.1%}")
    c4.metric("Recall", f"{m['recall']:.1%}")
    c5.metric("Est. Scenario Cost", f"${m['estimated_cost']:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"], name="Precision"))
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"], name="Recall"))
        fig.add_vline(x=threshold, line_dash="dash", line_color="gray")
        fig.update_layout(title="Precision / Recall vs. Threshold", xaxis_title="Threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cost_curve["threshold"], y=cost_curve["total_cost"], name="Total scenario cost"))
        fig.add_vline(x=threshold, line_dash="dash", line_color="gray")
        best_t = summary.get("cost_optimal_threshold")
        if best_t is not None:
            fig.add_vline(x=best_t, line_dash="dot", line_color="green",
                          annotation_text="cost-optimal")
        fig.update_layout(title="Estimated Scenario Cost vs. Threshold", xaxis_title="Threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(f"Fixed Investigation Capacity ({DAILY_REVIEW_CAPACITY} alerts/day)")
    daily = load_daily_capacity()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fraud_in_day"], name="Fraud in day"))
    fig.add_trace(go.Bar(x=daily["day"], y=daily["fraud_captured"], name="Fraud captured @ capacity"))
    fig.update_layout(barmode="overlay", title="Daily Fraud Capture Under a Fixed Review Capacity")
    st.plotly_chart(fig, use_container_width=True)
    avg_recall = daily["recall"].mean()
    st.caption(
        f"Averaged across the test period, a {DAILY_REVIEW_CAPACITY}-alert/day "
        f"capacity captures about **{avg_recall:.1%}** of daily fraud."
    )


def main():
    scored = load_scored()
    summary = load_summary()

    st.sidebar.title("Fraud Risk & Investigation Console")
    page = st.sidebar.radio(
        "Page", ["Fraud Overview", "Alert Queue", "Transaction Investigation", "Threshold Simulator"],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Data: IEEE-CIS Fraud Detection (Kaggle competition, Vesta Corporation). "
        "Real anonymized transactions; many fields are intentionally masked by "
        "the provider and are never assigned invented meanings here."
    )

    if page == "Fraud Overview":
        page_overview(scored, summary)
    elif page == "Alert Queue":
        page_alert_queue(scored)
    elif page == "Transaction Investigation":
        bundle = load_model_bundle()
        page_investigation(scored, bundle)
    elif page == "Threshold Simulator":
        sweep = load_threshold_sweep()
        cost_curve = load_cost_curve()
        page_threshold_simulator(scored, summary, sweep, cost_curve)


if __name__ == "__main__":
    main()
