"""
Automated Test Suite for Churn Intelligence Pipeline
===================================================
Tests data ingestion, SQL execution, machine learning training,
SHAP explainability calculations, and the retention simulation engine.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import run_full_data_pipeline, get_db_connection, PROCESSED_DATA_PATH
from src.eda_analysis import compute_executive_kpis, run_categorical_statistical_tests, run_numerical_statistical_tests
from src.churn_model import prepare_datasets, train_and_compare_models, compute_and_save_shap, MODEL_FILE, SHAP_DATA_FILE
from src.retention_strategy import simulate_retention_intervention, load_model_and_artifacts


def test_01_data_ingestion_and_cleaning():
    """Verify data loads cleanly without unexpected nulls and DuckDB is populated."""
    run_full_data_pipeline()
    assert os.path.exists(PROCESSED_DATA_PATH), "Processed CSV file was not created"
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    assert len(df) > 0, "Dataset is empty"
    assert "TotalCharges" in df.columns, "TotalCharges column missing"
    assert df["TotalCharges"].isna().sum() == 0, "TotalCharges has unresolved null values"
    assert "ChurnNumeric" in df.columns, "ChurnNumeric target column missing"
    
    # Test DuckDB connection and query
    con = get_db_connection()
    count_res = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    assert count_res == len(df), "DuckDB record count does not match CSV count"


def test_02_eda_and_statistical_testing():
    """Verify EDA KPIs and statistical hypothesis tests compute valid metrics."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    kpis = compute_executive_kpis(df)
    
    assert 0 < kpis["churn_rate_pct"] < 100, "Churn rate percentage out of range"
    assert kpis["total_mrr"] > 0, "Total MRR should be positive"
    assert kpis["churned_mrr"] > 0, "Churned MRR should be positive"
    
    cat_tests = run_categorical_statistical_tests(df)
    assert len(cat_tests) > 0, "Categorical statistical tests returned empty"
    assert "Contract" in cat_tests["Feature"].values, "Contract should be tested"
    
    num_tests = run_numerical_statistical_tests(df)
    assert len(num_tests) > 0, "Numerical statistical tests returned empty"


def test_03_ml_and_shap_pipeline():
    """Verify ML model training, benchmarking, and SHAP explainability generation."""
    data_dict = prepare_datasets()
    results, best_model_name = train_and_compare_models(data_dict)
    
    assert best_model_name in results, "Best model not found in benchmark results"
    best_roc_auc = results[best_model_name]["roc_auc"]
    assert best_roc_auc >= 0.80, f"Model ROC-AUC ({best_roc_auc}) should meet performance target >= 0.80"
    
    best_model = results[best_model_name]["model"]
    shap_artifacts = compute_and_save_shap(data_dict, best_model)
    
    assert os.path.exists(MODEL_FILE), "Model file not saved"
    assert os.path.exists(SHAP_DATA_FILE), "SHAP artifacts file not saved"
    assert len(shap_artifacts["feature_importance_df"]) > 0, "SHAP feature importance dataframe is empty"


def test_04_retention_simulator():
    """Verify What-If retention simulation engine recalculates risk and financial impact."""
    artifacts = load_model_and_artifacts()
    sample_cust = artifacts["df"].iloc[0].to_dict()
    
    sim_result = simulate_retention_intervention(
        original_customer_data=sample_cust,
        new_contract="Two year",
        new_tech_support="Yes",
        discount_pct=10.0,
        model=artifacts["model"],
        preprocessor=artifacts["preprocessor"]
    )
    
    assert "simulated_churn_prob" in sim_result
    assert "risk_reduction_pct" in sim_result
    assert "net_annual_financial_gain" in sim_result
    assert sim_result["simulated_churn_prob"] <= sim_result["base_churn_prob"], "Upgraded contract should reduce churn risk"


def test_05_sql_queries_execution():
    """Verify that all queries in sql/02_kpi_and_cohort_analysis.sql execute without syntax errors."""
    sql_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sql", "02_kpi_and_cohort_analysis.sql")
    assert os.path.exists(sql_path), "SQL file does not exist"
    
    con = get_db_connection()
    with open(sql_path, "r") as f:
        sql_content = f.read()
    
    # Split queries by semicolon
    queries = [q.strip() for q in sql_content.split(";") if q.strip() and not q.strip().startswith("--")]
    
    for q in queries:
        df_res = con.execute(q).fetchdf()
        assert len(df_res) > 0, f"Query returned 0 rows: {q[:50]}..."
