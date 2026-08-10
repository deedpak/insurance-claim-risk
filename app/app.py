"""
Insurance Claim Risk Detection - Streamlit App
Loads the model trained in notebook 04, scores uploaded claims,
explains predictions with SHAP, and generates a plain-English
investigator summary using an LLM (GitHub Models - free with a GitHub account).
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import requests

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Insurance Claim Risk Detection", layout="wide")
st.title("🔍 Insurance Claim Risk Detection")
st.caption("Upload a claims CSV to score fraud risk, see why each claim was flagged, "
           "and get an investigator-friendly AI explanation.")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ---------------------------------------------------------------------------
# Load model + metadata (cached so it only loads once, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    categorical_columns = joblib.load(os.path.join(MODEL_DIR, "categorical_columns.pkl"))
    return model, feature_columns, categorical_columns


try:
    model, feature_columns, categorical_columns = load_model_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found in the `models/` folder. "
        "Run notebook 04_modeling_explainability.ipynb all the way to the end first — "
        "it saves fraud_model.pkl, feature_columns.pkl, and categorical_columns.pkl."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Feature engineering — MUST mirror notebook 03 exactly, or the model
# will get columns it doesn't recognise.
# ---------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    if 'incident_date' in df.columns and 'policy_bind_date' in df.columns:
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], errors='coerce')
        df['policy_age_days'] = (df['incident_date'] - df['policy_bind_date']).dt.days

    if 'months_as_customer' in df.columns:
        df['customer_tenure_years'] = df['months_as_customer'] / 12
        if 'total_claim_amount' in df.columns:
            df['claim_amount_per_tenure_year'] = df['total_claim_amount'] / \
                df['customer_tenure_years'].replace(0, 0.1)

    if 'total_claim_amount' in df.columns:
        threshold = df['total_claim_amount'].quantile(0.90)
        df['is_high_value_claim'] = (df['total_claim_amount'] >= threshold).astype(int)

    for part, total_col in [('vehicle_claim', 'total_claim_amount'),
                             ('injury_claim', 'total_claim_amount'),
                             ('property_claim', 'total_claim_amount')]:
        if part in df.columns and total_col in df.columns:
            df[f'{part}_ratio'] = df[part] / df[total_col].replace(0, np.nan)

    if all(c in df.columns for c in ['police_report_available', 'witnesses', 'property_damage']):
        df['missing_doc_count'] = (
            (df['police_report_available'].isin(['NO', 'Unknown', '?'])).astype(int) +
            (df['witnesses'] == 0).astype(int) +
            (df['property_damage'].isin(['NO', 'Unknown', '?'])).astype(int)
        )

    if 'incident_hour_of_the_day' in df.columns:
        df['is_odd_hour_incident'] = df['incident_hour_of_the_day'].apply(
            lambda h: 1 if pd.notnull(h) and (h <= 5 or h >= 22) else 0
        )

    return df


def prepare_for_model(df_raw):
    """Clean, engineer, encode, and align columns to exactly match what the model expects."""
    df = df_raw.replace('?', np.nan)
    df_feat = engineer_features(df)

    drop_cols = ['fraud_reported', 'fraud_flag', 'incident_date', 'policy_bind_date',
                 'incident_location', 'policy_number', 'insured_zip', '_c39']
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors='ignore')

    cat_cols_present = [c for c in categorical_columns if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=cat_cols_present, drop_first=True)

    # Align to the exact columns the model was trained on:
    # add any missing dummy columns as 0, drop any extras, keep correct order.
    X_aligned = X_encoded.reindex(columns=feature_columns, fill_value=0)

    return df_feat, X_aligned


# ---------------------------------------------------------------------------
# LLM explanation (GitHub Models - free tier via GitHub account, no paid API key)
# Falls back to a clear template-based explanation if no token is configured,
# so the app still works end-to-end even without the AI step set up.
# ---------------------------------------------------------------------------
def generate_ai_explanation(claim_summary: dict, top_factors: list) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "")

    factors_text = "; ".join([f"{f['feature']} (impact: {f['impact']:+.3f})" for f in top_factors])

    prompt = (
        "You are assisting an insurance investigator. Based ONLY on the structured claim data "
        "and model output below, write a short (2-3 sentence) plain-English explanation of why "
        "this claim received its risk score. Do not decide whether it is fraud - only explain the "
        "signals. Do not invent any facts not given below.\n\n"
        f"Risk probability: {claim_summary['risk_probability']:.0%}\n"
        f"Risk category: {claim_summary['risk_category']}\n"
        f"Most influential factors (feature, SHAP impact): {factors_text}\n"
        f"Claim amount: {claim_summary.get('total_claim_amount', 'N/A')}\n"
        f"Incident severity: {claim_summary.get('incident_severity', 'N/A')}\n"
    )

    if not api_key:
        # Fallback: deterministic template, no external call. Keeps the app fully functional.
        factor_names = ", ".join([f['feature'] for f in top_factors[:3]])
        return (
            f"This claim was classified as **{claim_summary['risk_category']} risk** "
            f"({claim_summary['risk_probability']:.0%} probability) primarily due to: {factor_names}. "
            f"(Set the GEMINI_API_KEY environment variable to enable live AI-generated explanations "
            f"via Google Gemini — currently showing a rule-based summary.)"
        )

    try:
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        factor_names = ", ".join([f['feature'] for f in top_factors[:3]])
        return (
            f"[AI service unavailable, showing rule-based summary — {e}] "
            f"This claim was classified as {claim_summary['risk_category']} risk "
            f"({claim_summary['risk_probability']:.0%} probability) primarily due to: {factor_names}."
        )


def risk_category(prob):
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Sidebar: upload
# ---------------------------------------------------------------------------
st.sidebar.header("1. Upload Claims Data")
uploaded_file = st.sidebar.file_uploader("Upload a claims CSV", type=["csv"])

if uploaded_file is None:
    st.info("👈 Upload a CSV to get started. You can use `data/insurance_claims.csv` "
            "(the original dataset) to try the app out.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

# ---------------------------------------------------------------------------
# Data quality section
# ---------------------------------------------------------------------------
st.header("📋 Data Quality Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total claims", len(df_raw))
col2.metric("Columns", df_raw.shape[1])
missing_pct = (df_raw.replace('?', np.nan).isnull().sum().sum() /
               (df_raw.shape[0] * df_raw.shape[1]) * 100)
col3.metric("Missing values", f"{missing_pct:.1f}%")
col4.metric("Duplicate rows", int(df_raw.duplicated().sum()))

with st.expander("View raw data sample"):
    st.dataframe(df_raw.head(20))

# ---------------------------------------------------------------------------
# Score all claims
# ---------------------------------------------------------------------------
st.header("🎯 Scoring Claims")

with st.spinner("Scoring claims..."):
    df_feat, X_aligned = prepare_for_model(df_raw)
    probs = model.predict_proba(X_aligned)[:, 1]
    df_feat['risk_probability'] = probs
    df_feat['risk_category'] = df_feat['risk_probability'].apply(risk_category)

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X_aligned)
    if isinstance(raw_shap, list):
        shap_vals = raw_shap[1]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_vals = raw_shap[:, :, 1]
    else:
        shap_vals = raw_shap

st.success(f"Scored {len(df_feat)} claims.")

# ---------------------------------------------------------------------------
# Model performance (only if ground-truth labels are present in the upload)
# ---------------------------------------------------------------------------
if 'fraud_reported' in df_raw.columns:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    y_true = df_raw['fraud_reported'].map({'Y': 1, 'N': 0})
    y_pred = (df_feat['risk_probability'] >= 0.5).astype(int)

    st.header("📊 Model Performance (ground-truth labels found in upload)")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2f}")
    p2.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2f}")
    p3.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.2f}")
    p4.metric("ROC-AUC", f"{roc_auc_score(y_true, df_feat['risk_probability']):.2f}")

# ---------------------------------------------------------------------------
# Filter high-risk claims
# ---------------------------------------------------------------------------
st.header("🚩 Review Claims")
risk_filter = st.multiselect(
    "Filter by risk category", options=["High", "Medium", "Low"], default=["High", "Medium"]
)
filtered = df_feat[df_feat['risk_category'].isin(risk_filter)].sort_values(
    'risk_probability', ascending=False
)

display_cols = [c for c in ['risk_probability', 'risk_category', 'total_claim_amount',
                             'incident_severity', 'incident_type'] if c in filtered.columns]
st.dataframe(
    filtered[display_cols].style.format({'risk_probability': '{:.1%}'}),
    use_container_width=True
)

# ---------------------------------------------------------------------------
# Individual claim explanation
# ---------------------------------------------------------------------------
st.header("🔬 Inspect an Individual Claim")

if len(filtered) == 0:
    st.warning("No claims match the current filter.")
else:
    selected_idx = st.selectbox(
        "Select a claim (shown by row number)",
        options=filtered.index.tolist(),
        format_func=lambda i: f"Row {i} — {df_feat.loc[i, 'risk_category']} risk "
                               f"({df_feat.loc[i, 'risk_probability']:.0%})"
    )

    row = df_feat.loc[selected_idx]
    position_in_full = df_feat.index.get_loc(selected_idx)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Fraud Risk Probability", f"{row['risk_probability']:.1%}")
        st.metric("Risk Category", row['risk_category'])
        if 'total_claim_amount' in row:
            st.metric("Claim Amount", f"${row['total_claim_amount']:,.0f}")

    with c2:
        st.subheader("Most influential factors")
        claim_shap = shap_vals[position_in_full]
        contrib = pd.DataFrame({
            'feature': X_aligned.columns,
            'impact': claim_shap
        }).sort_values('impact', key=abs, ascending=False).head(6)

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#C44E52' if v > 0 else '#4C72B0' for v in contrib['impact']]
        ax.barh(contrib['feature'], contrib['impact'], color=colors)
        ax.set_xlabel("Impact on risk score (red = increases risk)")
        ax.invert_yaxis()
        st.pyplot(fig)

    st.subheader("🤖 AI-generated investigator summary")
    top_factors = contrib.to_dict('records')
    claim_summary = {
        'risk_probability': row['risk_probability'],
        'risk_category': row['risk_category'],
        'total_claim_amount': row.get('total_claim_amount', 'N/A'),
        'incident_severity': row.get('incident_severity', 'N/A'),
    }
    with st.spinner("Generating explanation..."):
        explanation = generate_ai_explanation(claim_summary, top_factors)
    st.info(explanation)
    st.caption("⚠️ This explanation is AI-generated from model output only. "
               "It is NOT a fraud determination — a human investigator makes that call.")

# ---------------------------------------------------------------------------
# Download scored results
# ---------------------------------------------------------------------------
st.header("⬇️ Download Results")
output_cols = [c for c in df_raw.columns if c in df_feat.columns] + ['risk_probability', 'risk_category']
csv_buffer = io.StringIO()
df_feat[output_cols].to_csv(csv_buffer, index=False)
st.download_button(
    "Download scored claims as CSV",
    data=csv_buffer.getvalue(),
    file_name="scored_claims.csv",
    mime="text/csv"
)
