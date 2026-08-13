-- ============================================================================
-- SCRIPT 02: KPI, Cohort & Business Churn Analysis
-- Description: Advanced analytical queries using CTEs, Window Functions,
--              and Group By aggregations for executive reporting.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY 1: Executive KPI Summary (Macro Level)
-- Purpose: Quick executive snapshot of total customer base, revenue, and churn.
-- ----------------------------------------------------------------------------
SELECT 
    COUNT(customerID) AS total_customers,
    SUM(CASE WHEN ChurnNumeric = 1 THEN 1 ELSE 0 END) AS total_churned_customers,
    ROUND(100.0 * AVG(ChurnNumeric), 2) AS overall_churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_monthly_revenue_mrr,
    ROUND(SUM(MonthlyRevenueAtRisk), 2) AS monthly_revenue_lost_to_churn,
    ROUND(100.0 * SUM(MonthlyRevenueAtRisk) / SUM(MonthlyCharges), 2) AS revenue_at_risk_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_revenue_per_user_arpu
FROM customers;


-- ----------------------------------------------------------------------------
-- QUERY 2: Churn & Revenue Loss by Contract Type
-- Purpose: Discover which contract types carry the greatest attrition risk.
-- ----------------------------------------------------------------------------
SELECT 
    Contract,
    COUNT(customerID) AS total_accounts,
    SUM(ChurnNumeric) AS churned_accounts,
    ROUND(100.0 * AVG(ChurnNumeric), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_mrr,
    ROUND(SUM(MonthlyRevenueAtRisk), 2) AS mrr_lost_to_churn,
    ROUND(AVG(tenure), 1) AS avg_tenure_months
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 3: Tenure Cohort Retention Analysis
-- Purpose: Track how customer retention changes as tenure increases.
-- ----------------------------------------------------------------------------
SELECT 
    TenureCohort,
    COUNT(customerID) AS total_customers,
    SUM(ChurnNumeric) AS churned_customers,
    ROUND(100.0 * AVG(ChurnNumeric), 2) AS churn_rate_pct,
    ROUND(100.0 * (1 - AVG(ChurnNumeric)), 2) AS retention_rate_pct,
    ROUND(SUM(MonthlyCharges), 2) AS total_cohort_mrr,
    ROUND(SUM(MonthlyRevenueAtRisk), 2) AS cohort_mrr_lost
FROM customers
GROUP BY TenureCohort
ORDER BY TenureCohort;


-- ----------------------------------------------------------------------------
-- QUERY 4: Payment Method Friction & Churn Correlation
-- Purpose: Check if manual/electronic check methods have higher churn than autopay.
-- ----------------------------------------------------------------------------
SELECT 
    PaymentMethod,
    COUNT(customerID) AS customer_count,
    ROUND(100.0 * AVG(ChurnNumeric), 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill,
    ROUND(SUM(MonthlyRevenueAtRisk), 2) AS total_mrr_lost
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 5: Service Add-On Protection (TechSupport & OnlineSecurity)
-- Purpose: Quantify the churn reduction impact of providing tech support.
-- ----------------------------------------------------------------------------
SELECT 
    TechSupport,
    OnlineSecurity,
    COUNT(customerID) AS customer_count,
    ROUND(100.0 * AVG(ChurnNumeric), 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
WHERE InternetService != 'No'
GROUP BY TechSupport, OnlineSecurity
ORDER BY churn_rate_pct DESC;


-- ----------------------------------------------------------------------------
-- QUERY 6: Window Function - High-Value At-Risk Customer Priority List
-- Purpose: Ranks all churned/at-risk customers by Monthly Charges and calculates
--          the cumulative revenue loss to prioritize high-value accounts.
-- ----------------------------------------------------------------------------
WITH AtRiskAccounts AS (
    SELECT 
        customerID,
        Contract,
        PaymentMethod,
        tenure,
        MonthlyCharges,
        TotalCharges,
        -- Calculate rank within their contract type by spend
        DENSE_RANK() OVER (
            PARTITION BY Contract 
            ORDER BY MonthlyCharges DESC
        ) as rank_in_contract_tier,
        -- Calculate cumulative revenue within contract tier
        SUM(MonthlyCharges) OVER (
            PARTITION BY Contract 
            ORDER BY MonthlyCharges DESC 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) as running_tier_mrr
    FROM customers
    WHERE ChurnNumeric = 1
)
SELECT 
    customerID,
    Contract,
    PaymentMethod,
    tenure AS tenure_months,
    MonthlyCharges AS monthly_bill,
    TotalCharges AS lifetime_spend,
    rank_in_contract_tier,
    ROUND(running_tier_mrr, 2) AS cumulative_mrr_lost_in_tier
FROM AtRiskAccounts
WHERE rank_in_contract_tier <= 5
ORDER BY Contract, rank_in_contract_tier;
