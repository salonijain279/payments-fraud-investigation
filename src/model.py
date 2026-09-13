"""Train, evaluate, and analyze the fraud model.

Pipeline:
  1. Chronological train/test split on TransactionDT (first 80% -> train,
     last 20% -> test) -- NOT a random split, so the model is evaluated on
     transactions strictly *after* everything it trained on, which is the
     realistic way to validate a fraud model meant to score future traffic.
  2. Baseline: Logistic Regression on the interpretable feature set only.
  3. Main model: XGBoost on the full feature set (interpretable + masked
     C/D/M/V/id_ columns as opaque inputs), using native categorical support
     and native missing-value handling.
  4. Threshold sweep (precision/recall/alert volume at each cutoff).
  5. Fixed-capacity analysis: Precision@K / Fraud-Capture@K for a
     500-alert/day investigator review capacity, simulated per day across
     the test period.
  6. Cost-sensitive threshold: minimizes an assumed cost function
     (false negative = $500, false positive review cost = $10 -- explicitly
     scenario assumptions, not real industry costs).
  7. SHAP global feature importance + per-transaction local explanations.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (
    INTERPRETABLE_MODEL_FEATURES,
    build_features,
    get_model_feature_columns,
)

DATA_PATH = Path("data/processed/transactions_features.parquet")
MODEL_PATH = Path("models/xgboost_fraud.pkl")
OUTPUTS_DIR = Path("outputs")

TRAIN_FRAC = 0.8
FALSE_NEGATIVE_COST = 500  # scenario assumption, not a real industry figure
FALSE_POSITIVE_COST = 10   # scenario assumption, not a real industry figure
DAILY_REVIEW_CAPACITY = 500


def chronological_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    train, test = df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
    print(
        f"Train: {len(train):,} rows (days {train['transaction_day'].min()}-{train['transaction_day'].max()}), "
        f"fraud rate {train['isFraud'].mean():.4%}"
    )
    print(
        f"Test:  {len(test):,} rows (days {test['transaction_day'].min()}-{test['transaction_day'].max()}), "
        f"fraud rate {test['isFraud'].mean():.4%}"
    )
    return train, test


def train_baseline_logreg(train: pd.DataFrame, test: pd.DataFrame):
    feature_cols = [c for c in INTERPRETABLE_MODEL_FEATURES if c in train.columns]
    cat_cols = [c for c in feature_cols if train[c].dtype.name in ("object", "category")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
            ]), cat_cols),
        ]
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])

    X_train, y_train = train[feature_cols], train["isFraud"]
    X_test, y_test = test[feature_cols], test["isFraud"]

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    print(f"[Logistic Regression baseline] ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")
    return pipe, proba, metrics


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    X_train, y_train = train[feature_cols], train["isFraud"]
    X_test, y_test = test[feature_cols], test["isFraud"]

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    print(f"[XGBoost] ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}")
    return model, proba, metrics


def metrics_at_threshold(y_true, proba, threshold: float) -> dict:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "alerts": int(preds.sum()),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def threshold_sweep(y_true, proba, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.1)
    rows = [metrics_at_threshold(y_true, proba, t) for t in thresholds]
    return pd.DataFrame(rows)


def capacity_at_k(y_true: np.ndarray, proba: np.ndarray, k: int) -> dict:
    order = np.argsort(-proba)[:k]
    top_y = np.asarray(y_true)[order]
    fraud_total = int(np.asarray(y_true).sum())
    captured = int(top_y.sum())
    return {
        "k": k,
        "fraud_captured": captured,
        "fraud_total": fraud_total,
        "precision_at_k": captured / k if k else 0.0,
        "recall_at_k": captured / fraud_total if fraud_total else 0.0,
    }


def daily_capacity_simulation(test: pd.DataFrame, proba: np.ndarray, capacity: int = DAILY_REVIEW_CAPACITY) -> pd.DataFrame:
    """Simulate investigators who can only review `capacity` alerts per day.

    Each day, the top-`capacity` scored transactions are "reviewed"; fraud
    outside that cutoff is missed that day. This is a more realistic
    operational view than a single global top-K cut, since it's bound by a
    per-day review capacity rather than a capacity over the whole test window.
    """
    t = test[["transaction_day", "isFraud"]].copy()
    t["fraud_probability"] = proba
    rows = []
    for day, grp in t.groupby("transaction_day"):
        grp_sorted = grp.sort_values("fraud_probability", ascending=False)
        reviewed = grp_sorted.head(capacity)
        rows.append({
            "day": day,
            "transactions": len(grp),
            "fraud_in_day": int(grp["isFraud"].sum()),
            "alerts_reviewed": len(reviewed),
            "fraud_captured": int(reviewed["isFraud"].sum()),
            "precision": reviewed["isFraud"].mean() if len(reviewed) else 0.0,
        })
    daily = pd.DataFrame(rows).sort_values("day")
    daily["recall"] = daily["fraud_captured"] / daily["fraud_in_day"].replace(0, np.nan)
    return daily


def cost_sensitive_threshold(y_true, proba, fn_cost=FALSE_NEGATIVE_COST, fp_cost=FALSE_POSITIVE_COST) -> pd.DataFrame:
    thresholds = np.arange(0.02, 0.98, 0.02)
    rows = []
    for t in thresholds:
        m = metrics_at_threshold(y_true, proba, t)
        total_cost = m["false_negatives"] * fn_cost + m["false_positives"] * fp_cost
        rows.append({**m, "total_cost": total_cost})
    return pd.DataFrame(rows)


def risk_tier(p: float) -> str:
    if p >= 0.8:
        return "Critical"
    elif p >= 0.6:
        return "High"
    elif p >= 0.3:
        return "Medium"
    return "Low"


def compute_shap(model: xgb.XGBClassifier, X: pd.DataFrame, sample_size: int = None):
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values, X_sample


def global_feature_importance(shap_values, feature_names: list) -> pd.DataFrame:
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    mean_abs = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def main():
    df = pd.read_parquet(DATA_PATH)
    train, test = chronological_split(df)

    feature_cols = get_model_feature_columns(df, include_masked=True)
    print(f"Training XGBoost on {len(feature_cols)} features "
          f"({len(INTERPRETABLE_MODEL_FEATURES)} interpretable + masked families)")

    logreg_pipe, logreg_proba, logreg_metrics = train_baseline_logreg(train, test)
    xgb_model, xgb_proba, xgb_metrics = train_xgboost(train, test, feature_cols)

    y_test = test["isFraud"].values

    OUTPUTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    # --- threshold sweep ---
    sweep = threshold_sweep(y_test, xgb_proba)
    sweep.to_csv(OUTPUTS_DIR / "threshold_sweep.csv", index=False)
    print("\nThreshold sweep:\n", sweep.round(4).to_string(index=False))

    # --- fixed capacity (single global top-K, as a headline number) ---
    cap500 = capacity_at_k(y_test, xgb_proba, k=DAILY_REVIEW_CAPACITY)
    print(f"\nFraud Capture@{DAILY_REVIEW_CAPACITY}: {cap500}")

    # --- realistic per-day capacity simulation ---
    daily = daily_capacity_simulation(test, xgb_proba, capacity=DAILY_REVIEW_CAPACITY)
    daily.to_csv(OUTPUTS_DIR / "daily_capacity_simulation.csv", index=False)

    # --- cost-sensitive threshold ---
    cost_curve = cost_sensitive_threshold(y_test, xgb_proba)
    cost_curve.to_csv(OUTPUTS_DIR / "cost_sensitive_threshold.csv", index=False)
    best_row = cost_curve.loc[cost_curve["total_cost"].idxmin()]
    print(f"\nCost-minimizing threshold: {best_row['threshold']:.2f} "
          f"(total scenario cost ${best_row['total_cost']:,.0f}, "
          f"precision={best_row['precision']:.3f}, recall={best_row['recall']:.3f})")

    # --- SHAP ---
    X_test = test[feature_cols]
    explainer, shap_values, X_sample = compute_shap(xgb_model, X_test, sample_size=20000)
    importance = global_feature_importance(shap_values, feature_cols)
    importance.to_csv(OUTPUTS_DIR / "feature_importance.csv", index=False)
    print("\nTop 15 features by mean |SHAP|:\n", importance.head(15).to_string(index=False))

    # --- persist artifacts ---
    import joblib

    joblib.dump(
        {
            "xgb_model": xgb_model,
            "logreg_pipe": logreg_pipe,
            "feature_cols": feature_cols,
            "interpretable_cols": [c for c in INTERPRETABLE_MODEL_FEATURES if c in df.columns],
            "cost_optimal_threshold": float(best_row["threshold"]),
        },
        MODEL_PATH,
    )
    print(f"\nSaved model bundle -> {MODEL_PATH}")

    summary = {
        "logreg": logreg_metrics,
        "xgboost": xgb_metrics,
        "fraud_capture_at_500": cap500,
        "cost_optimal_threshold": float(best_row["threshold"]),
        "cost_optimal_precision": float(best_row["precision"]),
        "cost_optimal_recall": float(best_row["recall"]),
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features": len(feature_cols),
    }
    with open(OUTPUTS_DIR / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model_summary.json -> {OUTPUTS_DIR / 'model_summary.json'}")

    return {
        "train": train, "test": test, "xgb_model": xgb_model, "xgb_proba": xgb_proba,
        "feature_cols": feature_cols, "explainer": explainer, "importance": importance,
    }


if __name__ == "__main__":
    main()
