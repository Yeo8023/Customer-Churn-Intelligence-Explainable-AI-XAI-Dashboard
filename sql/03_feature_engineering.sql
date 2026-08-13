-- ============================================================================
-- SCRIPT 03: Feature Engineering & Analytical Data View
-- Description: Transforms raw columns into engineered features and ratios
--              ready for predictive modeling and statistical analysis.
-- ============================================================================

CREATE OR REPLACE VIEW v_customer_features AS
SELECT 
    customerID,
    -- Target
    ChurnNumeric AS target_churn,
    
    -- Numerical Metrics & Ratios
    tenure,
    MonthlyCharges,
    TotalCharges,
    ROUND(TotalCharges / NULLIF(tenure, 0), 2) AS calculated_avg_monthly_spend,
    ROUND(MonthlyCharges / NULLIF(TotalCharges, 0), 4) AS monthly_to_total_ratio,
    
    -- Demographics
    CASE WHEN gender = 'Female' THEN 1 ELSE 0 END AS is_female,
    SeniorCitizen AS is_senior_citizen,
    CASE WHEN Partner = 'Yes' THEN 1 ELSE 0 END AS has_partner,
    CASE WHEN Dependents = 'Yes' THEN 1 ELSE 0 END AS has_dependents,
    
    -- Contract & Billing
    CASE WHEN Contract = 'Month-to-month' THEN 1 ELSE 0 END AS is_month_to_month,
    CASE WHEN Contract = 'One year' THEN 1 ELSE 0 END AS is_one_year_contract,
    CASE WHEN Contract = 'Two year' THEN 1 ELSE 0 END AS is_two_year_contract,
    CASE WHEN PaperlessBilling = 'Yes' THEN 1 ELSE 0 END AS has_paperless_billing,
    CASE WHEN PaymentMethod = 'Electronic check' THEN 1 ELSE 0 END AS uses_electronic_check,
    CASE WHEN PaymentMethod LIKE '%automatic%' THEN 1 ELSE 0 END AS uses_autopay,
    
    -- Core Services
    CASE WHEN PhoneService = 'Yes' THEN 1 ELSE 0 END AS has_phone_service,
    CASE WHEN MultipleLines = 'Yes' THEN 1 ELSE 0 END AS has_multiple_lines,
    CASE WHEN InternetService = 'Fiber optic' THEN 1 ELSE 0 END AS has_fiber_optic,
    CASE WHEN InternetService = 'DSL' THEN 1 ELSE 0 END AS has_dsl,
    CASE WHEN InternetService = 'No' THEN 1 ELSE 0 END AS has_no_internet,
    
    -- Value-Add Service Bundle Count (0 to 6 services)
    (
        (CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END) +
        (CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END)
    ) AS total_addon_services_count,

    -- Specific High-Impact Protections
    CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END AS has_tech_support,
    CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END AS has_online_security

FROM customers;
