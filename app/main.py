"""
AI-Powered Customer Churn Intelligence & Explainable AI (XAI) Dashboard
========================================================================
Main Streamlit application featuring Executive KPIs, Cohort Retention,
Individual AI Churn Risk Diagnoses with SHAP, and a What-If Retention Simulator.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root directory to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_db_connection, PROCESSED_DATA_PATH
from src.eda_analysis import compute_executive_kpis, run_categorical_statistical_tests
from src.retention_strategy import (
    load_model_and_artifacts,
    predict_single_customer_risk,
    simulate_retention_intervention
)

# Page configuration
st.set_page_config(
    page_title="AI Churn Intelligence & XAI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-danger {
        color: #f87171 !important;
    }
    .metric-success {
        color: #4ade80 !important;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .insight-box {
        background-color: #1e1b4b;
        border-left: 4px solid #818cf8;
        padding: 16px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Loads cleaned data and database connection."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df


@st.cache_resource
def load_ml_assets():
    """Loads trained ML model, preprocessor, and SHAP artifacts."""
    return load_model_and_artifacts()


def main():
    # Load dataset & ML models
    df = load_data()
    ml_bundle = load_ml_assets()

    st.title("🎯 AI Customer Churn Intelligence & Explainability Platform")
    st.caption("End-to-End Business Analytics, Predictive Risk Modeling & Explainable AI (SHAP)")

    # Sidebar Filter Controls
    st.sidebar.header("🔍 Global Filters")
    contract_filter = st.sidebar.multiselect(
        "Contract Type",
        options=df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist()
    )
    internet_filter = st.sidebar.multiselect(
        "Internet Service",
        options=df["InternetService"].unique().tolist(),
        default=df["InternetService"].unique().tolist()
    )
    payment_filter = st.sidebar.multiselect(
        "Payment Method",
        options=df["PaymentMethod"].unique().tolist(),
        default=df["PaymentMethod"].unique().tolist()
    )

    # Filter data
    filtered_df = df[
        (df["Contract"].isin(contract_filter)) &
        (df["InternetService"].isin(internet_filter)) &
        (df["PaymentMethod"].isin(payment_filter))
    ]

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive KPI Overview",
        "📈 Cohort & Driver Analytics",
        "🧠 AI Churn Inspector (SHAP)",
        "💡 Retention Strategy Simulator",
        "🗄️ SQL Analytics Console"
    ])

    # =========================================================================
    # TAB 1: EXECUTIVE KPI OVERVIEW
    # =========================================================================
    with tab1:
        st.subheader("Executive Macro KPIs & Revenue at Risk")
        kpis = compute_executive_kpis(filtered_df)

        # 4 Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Monitored Accounts</div>
                <div class="metric-value">{kpis['total_customers']:,}</div>
                <small style="color:#94a3b8">Active subscriptions in view</small>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Overall Churn Rate</div>
                <div class="metric-value metric-danger">{kpis['churn_rate_pct']}%</div>
                <small style="color:#f87171">{kpis['churned_customers']:,} accounts lost</small>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Monthly MRR</div>
                <div class="metric-value">${kpis['total_mrr']:,.2f}</div>
                <small style="color:#94a3b8">Avg ARPU: ${kpis['avg_arpu']}/mo</small>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Monthly MRR Lost to Churn</div>
                <div class="metric-value metric-danger">${kpis['churned_mrr']:,.2f}</div>
                <small style="color:#f87171">{kpis['mrr_loss_pct']}% of total revenue at risk</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row: Churn by Contract & Churn by Payment Method
        c_left, c_right = st.columns(2)
        with c_left:
            contract_summary = filtered_df.groupby("Contract").agg(
                Total=("customerID", "count"),
                ChurnRate=("ChurnNumeric", lambda x: round(x.mean() * 100, 1)),
                MRRLost=("MonthlyRevenueAtRisk", "sum")
            ).reset_index()

            fig_contract = px.bar(
                contract_summary,
                x="Contract",
                y="ChurnRate",
                text="ChurnRate",
                title="Churn Rate % by Contract Type",
                labels={"ChurnRate": "Churn Rate (%)"},
                color="ChurnRate",
                color_continuous_scale="Reds"
            )
            fig_contract.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_contract.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_contract, use_container_width=True)

        with c_right:
            payment_summary = filtered_df.groupby("PaymentMethod").agg(
                Total=("customerID", "count"),
                ChurnRate=("ChurnNumeric", lambda x: round(x.mean() * 100, 1))
            ).reset_index().sort_values(by="ChurnRate", ascending=True)

            fig_payment = px.bar(
                payment_summary,
                y="PaymentMethod",
                x="ChurnRate",
                orientation='h',
                text="ChurnRate",
                title="Churn Rate % by Payment Method",
                labels={"ChurnRate": "Churn Rate (%)"},
                color="ChurnRate",
                color_continuous_scale="Oranges"
            )
            fig_payment.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_payment.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig_payment, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <b>💡 Key Executive Insight:</b> Month-to-month contracts experience a dramatic <b>42.7% churn rate</b>, compared to just <b>2.8% for 2-Year contracts</b>. Additionally, customers paying via <b>Electronic Check</b> exhibit nearly 4x higher churn than those on automated payment methods.
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: COHORT & BEHAVIORAL ANALYTICS
    # =========================================================================
    with tab2:
        st.subheader("Tenure Cohort Dynamics & Service Add-On Protections")

        col_a, col_b = st.columns(2)
        with col_a:
            cohort_data = filtered_df.groupby("TenureCohort", observed=False).agg(
                TotalAccounts=("customerID", "count"),
                ChurnedAccounts=("ChurnNumeric", "sum"),
                ChurnRate=("ChurnNumeric", lambda x: round(x.mean() * 100, 1)),
                AvgMonthlyBill=("MonthlyCharges", "mean")
            ).reset_index()

            fig_cohort = px.line(
                cohort_data,
                x="TenureCohort",
                y="ChurnRate",
                markers=True,
                title="Customer Attrition Curve Across Tenure Cohorts",
                labels={"ChurnRate": "Churn Rate (%)", "TenureCohort": "Tenure Cohort"}
            )
            fig_cohort.update_traces(line_color="#38bdf8", line_width=4, marker_size=10)
            fig_cohort.update_layout(template="plotly_dark")
            st.plotly_chart(fig_cohort, use_container_width=True)

        with col_b:
            # Impact of Tech Support & Online Security
            support_data = filtered_df[filtered_df["InternetService"] != "No"].groupby(
                ["TechSupport", "OnlineSecurity"]
            ).agg(ChurnRate=("ChurnNumeric", lambda x: round(x.mean() * 100, 1))).reset_index()

            fig_heat = px.bar(
                support_data,
                x="TechSupport",
                y="ChurnRate",
                color="OnlineSecurity",
                barmode="group",
                title="Impact of Tech Support & Online Security on Churn",
                labels={"ChurnRate": "Churn Rate (%)"},
                color_discrete_sequence=["#f87171", "#4ade80"]
            )
            fig_heat.update_layout(template="plotly_dark")
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <b>💡 Cohort Takeaway:</b> The <b>First 12 Months (Year 1)</b> is the most critical window, accounting for over <b>47% of all churn events</b>. Customers who bundle <b>Tech Support + Online Security</b> have an 81% lower churn rate than customers with no protection add-ons.
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: AI CHURN INSPECTOR & SHAP EXPLAINABILITY
    # =========================================================================
    with tab3:
        st.subheader("Explainable AI (SHAP) - Individual Account Diagnosis")
        st.write("Understand the machine learning model's predictions down to the individual customer level.")

        # Benchmark badges
        model_name = ml_bundle["model_name"]
        st.info(f"Active Production Model: **{model_name}** | Model Explainability Engine: **TreeSHAP**")

        # Select customer from test dataset
        shap_data = ml_bundle["shap_artifacts"]
        test_ids = list(shap_data["test_customer_ids"])

        selected_id = st.selectbox("Select Customer Account ID to Diagnose:", options=test_ids[:100])
        idx = test_ids.index(selected_id)

        # Get raw customer row
        cust_row = df[df["customerID"] == selected_id].iloc[0].to_dict()
        model = ml_bundle["model"]
        preprocessor = ml_bundle["preprocessor"]

        churn_prob = predict_single_customer_risk(cust_row, model, preprocessor)
        risk_pct = round(churn_prob * 100, 1)

        # Risk indicator display
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Churn Probability", f"{risk_pct}%", delta=f"{risk_pct - 26.5:.1f}% vs baseline", delta_color="inverse")
        r2.metric("Contract Type", cust_row["Contract"])
        r3.metric("Tenure", f"{cust_row['tenure']} months")
        r4.metric("Monthly Bill", f"${cust_row['MonthlyCharges']:.2f}")

        # Local SHAP Explanation Chart
        st.markdown("#### 🔬 Why is this customer predicted at this risk level? (SHAP Feature Contributions)")
        
        # Calculate feature contributions for this sample
        sample_shap = shap_data["shap_values"].values[idx]
        feature_names = ml_bundle["shap_artifacts"]["X_test_proc"].columns.tolist()

        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP_Value": sample_shap
        }).sort_values(by="SHAP_Value", key=abs, ascending=False).head(8)

        contrib_df["Direction"] = np.where(contrib_df["SHAP_Value"] > 0, "Increases Churn Risk (+)", "Reduces Churn Risk (-)")
        contrib_df["Color"] = np.where(contrib_df["SHAP_Value"] > 0, "#f87171", "#4ade80")

        fig_shap = px.bar(
            contrib_df,
            x="SHAP_Value",
            y="Feature",
            orientation="h",
            color="Direction",
            title=f"Top 8 Factors Driving Prediction for Account {selected_id}",
            color_discrete_map={"Increases Churn Risk (+)": "#f87171", "Reduces Churn Risk (-)": "#4ade80"}
        )
        fig_shap.update_layout(template="plotly_dark", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_shap, use_container_width=True)

    # =========================================================================
    # TAB 4: WHAT-IF RETENTION STRATEGY SIMULATOR
    # =========================================================================
    with tab4:
        st.subheader("💡 'What-If' Retention Strategy & Financial ROI Simulator")
        st.write("Test targeted retention offers on high-risk accounts to calculate probability reduction and net revenue saved.")

        sim_id = st.selectbox("Select Customer to Simulate Retention Intervention:", options=test_ids[:100], key="sim_cust")
        sim_cust = df[df["customerID"] == sim_id].iloc[0].to_dict()

        col_sim_ctrl, col_sim_res = st.columns([1, 1])

        with col_sim_ctrl:
            st.markdown("##### 🛠️ Choose Retention Interventions:")
            new_contract = st.selectbox(
                "Upgrade Contract Duration:",
                options=["(Keep Current)", "Month-to-month", "One year", "Two year"],
                index=2 if sim_cust["Contract"] == "Month-to-month" else 0
            )
            contract_val = None if new_contract == "(Keep Current)" else new_contract

            new_tech_support = st.selectbox(
                "Add Free Tech Support Package:",
                options=["(Keep Current)", "Yes", "No"],
                index=1 if sim_cust["TechSupport"] != "Yes" else 0
            )
            tech_val = None if new_tech_support == "(Keep Current)" else new_tech_support

            new_payment = st.selectbox(
                "Switch to Automatic Payment (Autopay):",
                options=["(Keep Current)", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=1 if "automatic" not in sim_cust["PaymentMethod"] else 0
            )
            payment_val = None if new_payment == "(Keep Current)" else new_payment

            discount_pct = st.slider("Apply Retention Discount (% off monthly bill):", 0, 30, 10, 5)

        with col_sim_res:
            st.markdown("##### 📊 Financial Simulation Outcome:")
            sim_result = simulate_retention_intervention(
                original_customer_data=sim_cust,
                new_contract=contract_val,
                new_tech_support=tech_val,
                new_payment_method=payment_val,
                discount_pct=discount_pct,
                model=ml_bundle["model"],
                preprocessor=ml_bundle["preprocessor"]
            )

            # Gauge comparison
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sim_result["simulated_churn_prob"],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "New Simulated Churn Risk (%)"},
                delta={'reference': sim_result["base_churn_prob"], 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#38bdf8"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(74, 222, 128, 0.2)"},
                        {'range': [30, 60], 'color': "rgba(251, 191, 36, 0.2)"},
                        {'range': [60, 100], 'color': "rgba(248, 113, 113, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': sim_result["base_churn_prob"]
                    }
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("Risk Reduction", f"-{sim_result['risk_reduction_pct']}%", delta="Lower Risk", delta_color="normal")
            m2.metric("Net Annual Value Saved", f"${sim_result['net_annual_financial_gain']:,.2f}", delta="Positive ROI" if sim_result['is_positive_roi'] else "Loss", delta_color="normal" if sim_result['is_positive_roi'] else "inverse")

    # =========================================================================
    # TAB 5: SQL ANALYTICS CONSOLE
    # =========================================================================
    with tab5:
        st.subheader("🗄️ Interactive SQL Query Workbench")
        st.write("Execute live SQL queries directly against the analytical DuckDB database table `customers`.")

        sample_queries = {
            "Top 5 Churned Customers by Monthly Bill": """
SELECT customerID, Contract, PaymentMethod, tenure, MonthlyCharges, TotalCharges
FROM customers
WHERE ChurnNumeric = 1
ORDER BY MonthlyCharges DESC
LIMIT 5;
            """.strip(),
            "Monthly Revenue at Risk by Internet Service": """
SELECT 
    InternetService,
    COUNT(customerID) AS total_users,
    ROUND(100.0 * AVG(ChurnNumeric), 2) AS churn_rate_pct,
    ROUND(SUM(MonthlyRevenueAtRisk), 2) AS mrr_lost_to_churn
FROM customers
GROUP BY InternetService
ORDER BY mrr_lost_to_churn DESC;
            """.strip(),
            "Cohort Retention Summary": """
SELECT 
    TenureCohort,
    COUNT(customerID) AS total_customers,
    ROUND(100.0 * AVG(ChurnNumeric), 2) AS churn_rate_pct,
    ROUND(100.0 * (1 - AVG(ChurnNumeric)), 2) AS retention_rate_pct,
    ROUND(SUM(MonthlyRevenueAtRisk), 2) AS total_mrr_lost
FROM customers
GROUP BY TenureCohort
ORDER BY TenureCohort;
            """.strip()
        }

        query_choice = st.selectbox("Select a Sample Business Query or write custom SQL:", options=list(sample_queries.keys()))
        sql_input = st.text_area("SQL Query Editor:", value=sample_queries[query_choice], height=140)

        if st.button("▶ Run SQL Query", type="primary"):
            try:
                con = get_db_connection()
                query_result_df = con.execute(sql_input).fetchdf()
                st.success(f"Query returned {len(query_result_df)} rows successfully:")
                st.dataframe(query_result_df, use_container_width=True)
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")


if __name__ == "__main__":
    main()
