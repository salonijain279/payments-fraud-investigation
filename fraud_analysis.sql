-- Fraud pattern analysis over the IEEE-CIS Fraud Detection dataset, run via
-- DuckDB against data/processed/transactions_clean.parquet.
--
-- Run with:  duckdb -c ".read fraud_analysis.sql"
-- or from Python: duckdb.sql(open('fraud_analysis.sql').read())
--
-- NOTE: only columns with a documented real-world meaning (TransactionAmt,
-- ProductCD, card4/card6, addr, dist, email domains, DeviceType/DeviceInfo,
-- and time fields derived from TransactionDT) are referenced here. Masked
-- C/D/M/V/id_ columns are used as opaque model inputs elsewhere (src/features.py)
-- but are not given invented interpretations in this business-facing SQL.

CREATE OR REPLACE VIEW transactions AS
    SELECT * FROM read_parquet('data/processed/transactions_clean.parquet');

-- 1. Overall volume and fraud rate --------------------------------------------------
SELECT
    COUNT(*)                                          AS total_transactions,
    SUM(isFraud)                                       AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)                       AS fraud_rate_pct,
    ROUND(SUM(TransactionAmt * isFraud), 2)            AS total_fraud_value_usd,
    ROUND(AVG(TransactionAmt) FILTER (WHERE isFraud=1), 2) AS avg_fraud_amount,
    ROUND(MEDIAN(TransactionAmt) FILTER (WHERE isFraud=1), 2) AS median_fraud_amount,
    ROUND(AVG(TransactionAmt) FILTER (WHERE isFraud=0), 2) AS avg_legit_amount
FROM transactions;

-- 2. Fraud rate by product code ------------------------------------------------------
SELECT
    ProductCD,
    COUNT(*)                     AS transactions,
    SUM(isFraud)                 AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4) AS fraud_rate_pct,
    ROUND(AVG(TransactionAmt), 2) AS avg_amount
FROM transactions
GROUP BY ProductCD
ORDER BY fraud_rate_pct DESC;

-- 3. Fraud rate by card network / card type ------------------------------------------
SELECT
    COALESCE(card4, 'unknown') AS card_network,
    COALESCE(card6, 'unknown') AS card_type,
    COUNT(*)                     AS transactions,
    SUM(isFraud)                 AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4) AS fraud_rate_pct
FROM transactions
GROUP BY 1, 2
ORDER BY fraud_rate_pct DESC;

-- 4. Fraud rate by transaction amount band -------------------------------------------
SELECT
    CASE
        WHEN TransactionAmt < 25   THEN '<25'
        WHEN TransactionAmt < 100  THEN '25-99'
        WHEN TransactionAmt < 500  THEN '100-499'
        WHEN TransactionAmt < 2000 THEN '500-1999'
        ELSE '2000+'
    END AS amount_band,
    COUNT(*)                      AS transaction_count,
    SUM(isFraud)                  AS fraud_count,
    ROUND(AVG(isFraud) * 100, 4)  AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY MIN(TransactionAmt);

-- 5. Fraud rate by purchaser email domain (top 20 by volume) -------------------------
SELECT
    COALESCE(P_emaildomain, 'missing') AS purchaser_email_domain,
    COUNT(*)                      AS transactions,
    SUM(isFraud)                  AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)  AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY transactions DESC
LIMIT 20;

-- 6. Fraud rate by device type (mobile vs desktop, where identity data exists) -------
SELECT
    COALESCE(DeviceType, 'no identity data') AS device_type,
    COUNT(*)                      AS transactions,
    SUM(isFraud)                  AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)  AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY fraud_rate_pct DESC;

-- 7. Fraud rate by hour of day and day-of-week bucket --------------------------------
SELECT
    transaction_hour,
    COUNT(*)                      AS transactions,
    SUM(isFraud)                  AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)  AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY 1;

-- 8. Does having identity/device data at all correlate with fraud rate? -------------
SELECT
    CASE WHEN DeviceType IS NULL THEN 'no identity data' ELSE 'has identity data' END AS identity_present,
    COUNT(*)                      AS transactions,
    SUM(isFraud)                  AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)  AS fraud_rate_pct
FROM transactions
GROUP BY 1;

-- 9. High-value transactions (top 1% by amount): how much fraud do they hold? -------
WITH ranked AS (
    SELECT TransactionAmt, isFraud,
           PERCENT_RANK() OVER (ORDER BY TransactionAmt) AS pct_rank
    FROM transactions
)
SELECT
    COUNT(*) AS top_1pct_transactions,
    SUM(isFraud) AS fraud_in_top_1pct,
    ROUND(100.0 * SUM(isFraud) / (SELECT SUM(isFraud) FROM transactions), 2) AS pct_of_all_fraud_captured
FROM ranked
WHERE pct_rank >= 0.99;

-- 10. Purchaser vs recipient email domain mismatch -----------------------------------
SELECT
    CASE
        WHEN R_emaildomain IS NULL THEN 'no recipient email'
        WHEN P_emaildomain = R_emaildomain THEN 'same domain'
        ELSE 'different domain'
    END AS email_match_status,
    COUNT(*)                      AS transactions,
    SUM(isFraud)                  AS fraud_transactions,
    ROUND(AVG(isFraud) * 100, 4)  AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY fraud_rate_pct DESC;
