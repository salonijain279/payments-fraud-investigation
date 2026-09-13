# Digital Payments Fraud & Identity Risk Investigation Engine

**Python, SQL (DuckDB), XGBoost, SHAP, Streamlit**

## Problem

A payments/e-commerce fraud team receives far more transactions than investigators can
manually review. Fraud is rare (~3.5% of traffic here) and investigation capacity is
fixed — so the real product isn't "detect fraud," it's: **which transactions should an
investigator look at first, why, and what happens to fraud capture and false-positive
workload as the review threshold moves.**

## Solution

An end-to-end fraud analytics workflow: SQL-based exploratory analysis → interpretable +
masked feature engineering → an XGBoost risk model benchmarked against a logistic
regression baseline → threshold and cost-sensitive decisioning → SHAP-based reason codes
→ a four-page Streamlit investigation console.

## Dataset

**[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)**
(Kaggle competition, data provided by Vesta Corporation) — **real, anonymized**
e-commerce transactions, not synthetic. 590,540 labeled transactions
(`train_transaction.csv`, 394 columns) optionally joined to identity/device signals
(`train_identity.csv`, 41 columns, ~24% match rate) on `TransactionID`.

**Why this dataset over alternatives considered** (PaySim, Credit Card Fraud/ULB,
BankSim, Sparkov): it's the only one where the strongest predictors are *both*
high-performing *and* at least partially interpretable to a human investigator — email
domain, card network/type, device, product code, and derived transaction velocity, on
top of Vesta's masked engineered features. Anonymized-PCA datasets (Credit Card
Fraud/ULB) can't support investigator-facing reason codes at all, since every predictive
feature is an opaque principal component.

**Masked columns — a firm project rule:** `C1-C14` (counting), `D1-D15` (timedelta),
`M1-M9` (match flags), `V1-V339` (Vesta engineered features), and most `id_*` columns
have their *exact* real-world meaning masked by Vesta. This project uses them as opaque
numeric/categorical model inputs everywhere (feature engineering, SHAP output, dashboard
reason codes) and **never invents a specific interpretation for any individual masked
column** — they're always labeled generically by family (e.g. "masked counting feature
C14"), consistent with the competition's own data description.

Kaggle's `test_transaction.csv` / `test_identity.csv` have no fraud labels (they exist
only for the competition leaderboard) and are not used here. This project instead splits
its own chronological train/test set out of the 590,540 labeled rows.

## Architecture

```
Raw transaction + identity CSVs
        |
   SQL / DuckDB analysis (fraud_analysis.sql)
        |
   Feature engineering (src/features.py)
        |
   Chronological train/test split (src/model.py)
        |
   Logistic Regression baseline  +  XGBoost
        |
   Threshold sweep + fixed-capacity + cost-sensitive analysis
        |
   SHAP global importance + local reason codes (src/scoring.py)
        |
   Risk-scored transaction table (outputs/scored_transactions.csv)
        |
   Streamlit investigation console (dashboard/app.py)
```

## Repo structure

```
payments-fraud-investigation/
├── data/
│   ├── raw/            train_transaction.csv, train_identity.csv (not committed — see Setup)
│   └── processed/      cleaned + feature-engineered parquet files
├── notebooks/
│   ├── 01_eda.ipynb                 fraud rates, amount/time distributions, missingness
│   ├── 02_feature_engineering.ipynb interpretable + masked feature construction, sanity checks
│   └── 03_modeling.ipynb            baseline vs. XGBoost, thresholds, cost curve, SHAP
├── src/
│   ├── data_processing.py   load, merge, clean, downcast dtypes
│   ├── features.py          interpretable + masked-family feature engineering
│   ├── model.py              chronological split, both models, threshold/cost/SHAP analysis
│   └── scoring.py            final risk-scored table + SHAP reason codes
├── dashboard/
│   └── app.py                4-page Streamlit investigation console
├── models/xgboost_fraud.pkl
├── outputs/                   threshold_sweep.csv, cost_sensitive_threshold.csv,
│                               daily_capacity_simulation.csv, feature_importance.csv,
│                               scored_transactions.csv, model_summary.json
├── fraud_analysis.sql
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download the data yourself (requires accepting the Kaggle competition rules — this
project does not and cannot bypass that): from
[kaggle.com/competitions/ieee-fraud-detection/data](https://www.kaggle.com/competitions/ieee-fraud-detection/data),
get `train_transaction.csv` and `train_identity.csv` and place them in `data/raw/`.

Then run the pipeline end to end:

```bash
python src/data_processing.py   # -> data/processed/transactions_clean.parquet
python src/features.py          # -> data/processed/transactions_features.parquet
python src/model.py             # trains both models, writes outputs/ + models/xgboost_fraud.pkl
python src/scoring.py           # -> outputs/scored_transactions.csv
streamlit run dashboard/app.py
```

SQL analysis: `duckdb -c ".read fraud_analysis.sql"` (or run the queries in
`fraud_analysis.sql` from Python via `duckdb.sql(...)`).

## Key features

- **Chronological train/test split** (first 80% of transaction time → train, last 20% →
  test) instead of a random split — the model is validated the way it's actually used:
  scoring transactions it has never seen, that happen *after* everything it trained on.
- **Point-in-time behavioral/velocity features**, grouped by `card1` (a high-cardinality,
  low-missingness card-attribute code used here as a pseudo account identifier): prior
  transaction count, average/max prior amount, time since last transaction. Computed
  with expanding windows ordered by time, so no feature ever leaks future information.
- **Interpretable vs. masked feature separation**, carried through consistently from
  feature engineering into SHAP output and dashboard reason codes.
- **Threshold sweep, fixed-capacity simulation (Fraud Capture@K), and cost-sensitive
  threshold optimization** — turning a probability into an operational decision, not
  just a metric.
- **SHAP local + global explanations** feeding plain-English reason codes in the
  dashboard.

## Model evaluation (this run — chronological split, 472,432 train / 118,108 test rows)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (interpretable features only) | 0.757 | 0.143 |
| **XGBoost** (interpretable + masked features, 451 total) | **0.891** | **0.488** |

PR-AUC is the metric that matters here, not accuracy: fraud is ~3.5% of test
transactions, so a model that never alerts would score >96% accuracy while catching zero
fraud.

**Threshold trade-off** (full sweep in `outputs/threshold_sweep.csv`):

| Threshold | Alerts | Precision | Recall |
|---|---|---|---|
| 0.1 | 71,848 | 5.4% | 95.3% |
| 0.3 | 24,379 | 13.7% | 82.0% |
| 0.5 | 11,106 | 24.9% | 68.1% |
| 0.7 | 4,820 | 43.2% | 51.2% |
| 0.9 | 1,530 | 75.2% | 28.3% |

**Fixed investigation capacity** — assuming investigators can review 500 alerts/day,
simulated per day across the test period (`outputs/daily_capacity_simulation.csv`):
captures **~79.3%** of each day's fraud on average, at **~15.4%** average precision. A
single global top-500 cut over the whole test window (a less realistic, one-shot version
of the same idea) reaches 93.8% precision but only 11.5% recall, since it concentrates
review on the very highest-confidence cases only.

**Cost-sensitive threshold** — assuming (explicitly a scenario assumption, not a real
industry figure) a missed fraud costs $500 and a false-positive investigation costs $10:
the cost-minimizing threshold is **0.24** (precision 10.8%, recall 86.4%). At this
imbalance of assumed costs, the model correctly favors high recall — missing fraud is
25x more expensive than an unnecessary review.

**Top global SHAP drivers**: `P_emaildomain`, masked counting features `C14`/`C13`/`C1`,
`card6` (card type), `TransactionAmt`, and several masked Vesta features (`V70`, `V294`)
and identity signals (`id_31`). Full ranking in `outputs/feature_importance.csv`. Note
that several of the strongest predictors are masked — this is real, honestly reported
behavior of the dataset, not a modeling choice; interpretable features (email domain,
amount, card type, card velocity) are still consistently in the top 10.

## Limitations

- IEEE-CIS is real transaction data but from a single provider (Vesta) and a fixed
  ~6-month window in 2019 — model behavior will not directly transfer to a different
  payments product, geography, or era of fraud tactics.
- `TransactionDT` is a relative time offset, not a calendar timestamp — no real
  day-of-week/holiday seasonality can be derived from it beyond the mod-based buckets
  used here.
- Several of the model's strongest features are masked by the data provider; this
  project deliberately does not guess their meaning, which limits how much of the
  model's behavior can be explained to a human reviewer.
- Risk tiers (Critical/High/Medium/Low at 0.8/0.6/0.3) and the $500/$10 cost assumptions
  are project operating rules for this demo, not validated industry benchmarks.
- No real investigator feedback loop, no production transaction stream, and this
  project's model output indicates elevated risk warranting review — it is not a
  determination that a transaction is fraudulent.
