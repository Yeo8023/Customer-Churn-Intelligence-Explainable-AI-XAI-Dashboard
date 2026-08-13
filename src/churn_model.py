"""
Machine Learning & Explainable AI (SHAP) Churn Modeling
======================================================
This module trains predictive models (Logistic Regression, Random Forest, XGBoost),
evaluates performance, and generates SHAP (SHapley Additive exPlanations) values
for global and individual customer-level explainability.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
import shap

from src.data_loader import PROCESSED_DATA_PATH

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "best_churn_model.joblib")
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "preprocessor.joblib")
SHAP_DATA_FILE = os.path.join(MODEL_DIR, "shap_artifacts.joblib")


def get_feature_lists():
    """Returns lists of categorical and numerical features used for modeling."""
    cat_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    return cat_features, num_features


def prepare_datasets(test_size: float = 0.2, random_state: int = 42):
    """
    Loads clean data, splits into Train and Test sets, and builds preprocessor.
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)
    cat_features, num_features = get_feature_lists()

    X = df[cat_features + num_features].copy()
    y = df["ChurnNumeric"].values
    customer_ids = df["customerID"].values

    # Train / Test Split (Stratified to maintain churn ratio)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_features)
        ]
    )

    # Fit preprocessor on train data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    feature_names = num_features + encoded_cat_names

    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

    # Save preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_proc": X_train_df,
        "X_test_proc": X_test_df,
        "y_train": y_train,
        "y_test": y_test,
        "id_train": id_train,
        "id_test": id_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "full_df": df
    }


def train_and_compare_models(data_dict: dict) -> dict:
    """
    Trains and benchmarks Logistic Regression, Random Forest, and XGBoost.
    Returns the models, performance metrics, and the best-performing model.
    """
    X_train = data_dict["X_train_proc"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test_proc"]
    y_test = data_dict["y_test"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )
    }

    results = {}
    print("\n" + "=" * 60)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 60)

    best_model_name = None
    best_roc_auc = 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "y_pred_proba": y_pred_proba
        }

        print(f"\n[{name}]")
        print(f"  ROC-AUC Score: {roc_auc:.4f}")
        print(f"  PR-AUC Score:  {pr_auc:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name

    print(f"\n[WINNER] Selected '{best_model_name}' as the primary production model (ROC-AUC: {best_roc_auc:.4f})")
    best_model = results[best_model_name]["model"]

    # Save best model
    joblib.dump({
        "model": best_model,
        "name": best_model_name,
        "benchmark_results": {k: {m: v[m] for m in ["roc_auc", "pr_auc"]} for k, v in results.items()}
    }, MODEL_FILE)

    return results, best_model_name


def compute_and_save_shap(data_dict: dict, best_model) -> dict:
    """
    Computes TreeSHAP values on the test set for global and local explainability.
    Saves the precomputed values for high-speed dashboard visualization.
    """
    print("\n[INFO] Computing SHAP Explainability Values...")
    X_test_proc = data_dict["X_test_proc"]
    
    # Initialize TreeExplainer (or generic Explainer)
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer(X_test_proc)
    except Exception:
        # Fallback to Exact / Kernel explainer if needed
        explainer = shap.Explainer(best_model.predict_proba, X_test_proc.iloc[:100])
        shap_values = explainer(X_test_proc)

    # Extract base values and feature contributions
    if len(shap_values.shape) == 3:
        # For multi-output / binary classification proba, take class 1 (churn)
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values

    # Calculate global mean absolute SHAP importance
    mean_abs_shap = np.abs(shap_values_class1.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        "Feature": data_dict["feature_names"],
        "SHAP_Importance": mean_abs_shap
    }).sort_values(by="SHAP_Importance", ascending=False)

    shap_artifacts = {
        "explainer": explainer,
        "shap_values": shap_values_class1,
        "feature_importance_df": feature_importance_df,
        "test_customer_ids": data_dict["id_test"],
        "X_test_proc": X_test_proc,
        "X_test_raw": data_dict["X_test_raw"],
        "y_test": data_dict["y_test"]
    }

    joblib.dump(shap_artifacts, SHAP_DATA_FILE)
    print(f"[SUCCESS] SHAP artifacts saved to {SHAP_DATA_FILE}")
    print("\nTop 5 Global Churn Drivers (by SHAP):")
    print(feature_importance_df.head(5).to_string(index=False))

    return shap_artifacts


def run_full_modeling_pipeline():
    """Executes the complete data preparation, training, and SHAP pipeline."""
    print("=" * 60)
    print("STEP 2: Training Churn Prediction & Explainable AI Pipeline")
    print("=" * 60)
    data_dict = prepare_datasets()
    results, best_model_name = train_and_compare_models(data_dict)
    best_model = results[best_model_name]["model"]
    compute_and_save_shap(data_dict, best_model)
    print("=" * 60)
    print("Model Training & SHAP Pipeline Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_modeling_pipeline()
