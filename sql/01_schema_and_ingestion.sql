-- ============================================================================
-- SCRIPT 01: Database Schema & Ingestion Script
-- Description: Creates the customer churn table and indexes for fast queries.
-- ============================================================================

-- Drop existing table if refreshing
DROP TABLE IF EXISTS customers;

-- Create main analytical customers table
CREATE TABLE customers (
    customerID VARCHAR PRIMARY KEY,
    gender VARCHAR,
    SeniorCitizen INTEGER,
    Partner VARCHAR,
    Dependents VARCHAR,
    tenure INTEGER,
    PhoneService VARCHAR,
    MultipleLines VARCHAR,
    InternetService VARCHAR,
    OnlineSecurity VARCHAR,
    OnlineBackup VARCHAR,
    DeviceProtection VARCHAR,
    TechSupport VARCHAR,
    StreamingTV VARCHAR,
    StreamingMovies VARCHAR,
    Contract VARCHAR,
    PaperlessBilling VARCHAR,
    PaymentMethod VARCHAR,
    MonthlyCharges DOUBLE,
    TotalCharges DOUBLE,
    Churn VARCHAR,
    ChurnNumeric INTEGER,
    TenureCohort VARCHAR,
    MonthlyRevenueAtRisk DOUBLE
);

-- Note: When running with DuckDB / Python, data is populated directly
-- from cleaned CSV using:
-- INSERT INTO customers SELECT * FROM read_csv_auto('data/processed/telco_churn_clean.csv');
