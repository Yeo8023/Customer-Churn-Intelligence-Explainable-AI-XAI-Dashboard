"""
Exploratory Data Analysis (EDA) & Statistical Hypothesis Testing
===============================================================
This module analyzes customer churn patterns and mathematically tests which
factors significantly drive customer attrition using Chi-Square and T-tests.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import PROCESSED_DATA_PATH, get_db_connection

OUTPUT_FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")


def load_clean_data() -> pd.DataFrame:
    """Loads the processed customer churn dataset."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        from src.data_loader import run_full_data_pipeline
        run_full_data_pipeline()
    return pd.read_csv(PROCESSED_DATA_PATH)


def compute_executive_kpis(df: pd.DataFrame) -> dict:
    """
    Calculates top-line business metrics:
    - Total Customers & Churn Rate
    - Total Monthly Recurring Revenue (MRR)
    - Monthly Revenue Lost to Churn
    - Average Customer Lifetime (Tenure)
    """
    total_customers = len(df)
    churned_customers = int(df["ChurnNumeric"].sum())
    churn_rate = (churned_customers / total_customers) * 100.0
    
    total_mrr = float(df["MonthlyCharges"].sum())
    churned_mrr = float(df[df["ChurnNumeric"] == 1]["MonthlyCharges"].sum())
    mrr_loss_pct = (churned_mrr / total_mrr) * 100.0
    avg_arpu = float(df["MonthlyCharges"].mean())
    avg_tenure = float(df["tenure"].mean())

    kpis = {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "retained_customers": total_customers - churned_customers,
        "churn_rate_pct": round(churn_rate, 2),
        "total_mrr": round(total_mrr, 2),
        "churned_mrr": round(churned_mrr, 2),
        "mrr_loss_pct": round(mrr_loss_pct, 2),
        "avg_arpu": round(avg_arpu, 2),
        "avg_tenure_months": round(avg_tenure, 1)
    }
    return kpis


def run_categorical_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Chi-Square Test of Independence for all categorical features against Churn.
    P-value < 0.05 indicates statistically significant relationship with Churn.
    """
    cat_columns = [
        "Contract", "InternetService", "PaymentMethod", "TechSupport",
        "OnlineSecurity", "PaperlessBilling", "DeviceProtection",
        "OnlineBackup", "SeniorCitizen", "Partner", "Dependents", "PhoneService"
    ]

    results = []
    for col in cat_columns:
        if col in df.columns:
            # Contingency table
            contingency_table = pd.crosstab(df[col], df["Churn"])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
            
            # Cramér's V (Effect size measure: 0 to 1)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0

            results.append({
                "Feature": col,
                "Chi2_Statistic": round(chi2, 2),
                "P_Value": p_val,
                "Cramers_V_Effect_Size": round(cramers_v, 3),
                "Is_Statistically_Significant": p_val < 0.05
            })

    results_df = pd.DataFrame(results).sort_values(by="Chi2_Statistic", ascending=False)
    return results_df


def run_numerical_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Independent Two-Sample T-Tests comparing Churned vs Non-Churned groups
    across numerical features (Tenure, MonthlyCharges, TotalCharges).
    """
    num_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    results = []

    churned = df[df["ChurnNumeric"] == 1]
    retained = df[df["ChurnNumeric"] == 0]

    for col in num_columns:
        c_mean = churned[col].mean()
        r_mean = retained[col].mean()
        
        # T-Test (Welch's t-test assuming unequal variance)
        t_stat, p_val = stats.ttest_ind(churned[col], retained[col], equal_var=False)

        results.append({
            "Metric": col,
            "Churned_Mean": round(c_mean, 2),
            "Retained_Mean": round(r_mean, 2),
            "Difference": round(c_mean - r_mean, 2),
            "T_Statistic": round(t_stat, 2),
            "P_Value": p_val,
            "Is_Statistically_Significant": p_val < 0.05
        })

    return pd.DataFrame(results)


def generate_eda_summary_report() -> dict:
    """Runs all EDA steps and returns clean statistical outputs."""
    df = load_clean_data()
    kpis = compute_executive_kpis(df)
    cat_tests = run_categorical_statistical_tests(df)
    num_tests = run_numerical_statistical_tests(df)

    print("\n" + "=" * 60)
    print("EXECUTIVE CHURN KPIs")
    print("=" * 60)
    for k, v in kpis.items():
        print(f"  {k:30s}: {v}")

    print("\n" + "=" * 60)
    print("TOP CATEGORICAL CHURN DRIVERS (Chi-Square Test)")
    print("=" * 60)
    print(cat_tests.to_string(index=False))

    print("\n" + "=" * 60)
    print("NUMERICAL DIFFERENCES (Churned vs Retained T-Test)")
    print("=" * 60)
    print(num_tests.to_string(index=False))

    return {
        "kpis": kpis,
        "categorical_tests": cat_tests,
        "numerical_tests": num_tests
    }


if __name__ == "__main__":
    generate_eda_summary_report()
