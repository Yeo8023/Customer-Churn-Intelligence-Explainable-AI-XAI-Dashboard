"""
Retention Strategy & What-If Financial Simulation Engine
========================================================
This module evaluates business interventions for at-risk customers,
calculating risk reduction, projected revenue saved, and campaign ROI.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.churn_model import MODEL_FILE, PREPROCESSOR_FILE, SHAP_DATA_FILE
from src.data_loader import PROCESSED_DATA_PATH


def load_model_and_artifacts():
    """Loads the trained model, preprocessor, and precomputed SHAP data."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SHAP_DATA_FILE):
        from src.churn_model import run_full_modeling_pipeline
        run_full_modeling_pipeline()

    model_bundle = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    shap_artifacts = joblib.load(SHAP_DATA_FILE)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    return {
        "model": model_bundle["model"],
        "model_name": model_bundle["name"],
        "preprocessor": preprocessor,
        "shap_artifacts": shap_artifacts,
        "df": df
    }


def predict_single_customer_risk(customer_data: dict, model, preprocessor) -> float:
    """
    Given a dictionary of customer features, returns their predicted churn probability (0.0 to 1.0).
    """
    input_df = pd.DataFrame([customer_data])
    processed_input = preprocessor.transform(input_df)
    churn_proba = float(model.predict_proba(processed_input)[0, 1])
    return churn_proba


def simulate_retention_intervention(
    original_customer_data: dict,
    new_contract: str = None,
    new_tech_support: str = None,
    new_payment_method: str = None,
    discount_pct: float = 0.0,
    model=None,
    preprocessor=None
) -> dict:
    """
    Simulates business interventions on an individual customer:
    - Change contract duration (e.g. Month-to-month -> One year)
    - Add Tech Support service
    - Change payment method (e.g. Electronic check -> Bank transfer autopay)
    - Apply a monthly subscription discount
    
    Returns:
    - Base Churn Probability vs. New Simulated Churn Probability
    - Risk Reduction (%)
    - Annual Revenue at Risk before vs. after
    - Net Annual Financial Impact / Saved Revenue (taking discount into account)
    """
    if model is None or preprocessor is None:
        artifacts = load_model_and_artifacts()
        model = artifacts["model"]
        preprocessor = artifacts["preprocessor"]

    # Copy customer record
    simulated_data = dict(original_customer_data)

    # 1. Base prediction
    base_churn_prob = predict_single_customer_risk(original_customer_data, model, preprocessor)
    base_monthly_charges = float(original_customer_data["MonthlyCharges"])

    # 2. Apply modifications
    if new_contract:
        simulated_data["Contract"] = new_contract
    if new_tech_support:
        simulated_data["TechSupport"] = new_tech_support
    if new_payment_method:
        simulated_data["PaymentMethod"] = new_payment_method
    if discount_pct > 0.0:
        new_charges = round(base_monthly_charges * (1 - (discount_pct / 100.0)), 2)
        simulated_data["MonthlyCharges"] = new_charges

    # 3. New prediction
    simulated_churn_prob = predict_single_customer_risk(simulated_data, model, preprocessor)
    risk_reduction_pct = (base_churn_prob - simulated_churn_prob) * 100.0

    # 4. Financial Calculations
    new_monthly_charges = float(simulated_data["MonthlyCharges"])
    base_expected_annual_loss = base_churn_prob * base_monthly_charges * 12.0
    simulated_expected_annual_loss = simulated_churn_prob * new_monthly_charges * 12.0
    annual_discount_cost = (base_monthly_charges - new_monthly_charges) * 12.0

    gross_annual_saved_revenue = base_expected_annual_loss - simulated_expected_annual_loss
    net_annual_financial_gain = gross_annual_saved_revenue - annual_discount_cost

    return {
        "base_churn_prob": round(base_churn_prob * 100.0, 1),
        "simulated_churn_prob": round(simulated_churn_prob * 100.0, 1),
        "risk_reduction_pct": round(risk_reduction_pct, 1),
        "base_monthly_charges": base_monthly_charges,
        "new_monthly_charges": new_monthly_charges,
        "annual_discount_cost": round(annual_discount_cost, 2),
        "gross_annual_saved_revenue": round(gross_annual_saved_revenue, 2),
        "net_annual_financial_gain": round(net_annual_financial_gain, 2),
        "is_positive_roi": net_annual_financial_gain > 0
    }
