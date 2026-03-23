import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Survival Predictor",
    page_icon="🫀",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("survival_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

model, features = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫀 Heart Failure — Patient Survival Predictor")
st.markdown("""
**An explainable ML tool for clinical risk stratification in heart failure patients.**  
Built by [Afriyie Karikari Bempah, PharmD](https://linkedin.com/in/afriyiekarikaribempah) | 
[GitHub](https://github.com/akbempah1/patient-survival-prediction)

---
> ⚠️ *This tool is for educational and research purposes only.
> It does not constitute medical advice.*
""")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Patient Clinical Parameters")
st.sidebar.markdown("Adjust the values below to match patient data:")

age = st.sidebar.slider("Age (years)", 40, 95, 60)
ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 14, 80, 38)
serum_creatinine = st.sidebar.slider("Serum Creatinine (mg/dL)", 0.5, 9.4, 1.1)
serum_sodium = st.sidebar.slider("Serum Sodium (mEq/L)", 113, 148, 137)
creatinine_phosphokinase = st.sidebar.slider(
    "Creatinine Phosphokinase (mcg/L)", 23, 7861, 250)
platelets = st.sidebar.slider(
    "Platelets (kiloplatelets/mL)", 25100, 850000, 262000)

st.sidebar.markdown("---")
anaemia = st.sidebar.selectbox("Anaemia", ["No", "Yes"])
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", ["No", "Yes"])
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])

# ── Build input dataframe ─────────────────────────────────────────────────────
input_data = pd.DataFrame({
    'age': [age],
    'anaemia': [1 if anaemia == "Yes" else 0],
    'creatinine_phosphokinase': [creatinine_phosphokinase],
    'diabetes': [1 if diabetes == "Yes" else 0],
    'ejection_fraction': [ejection_fraction],
    'high_blood_pressure': [1 if high_blood_pressure == "Yes" else 0],
    'platelets': [platelets],
    'serum_creatinine': [serum_creatinine],
    'serum_sodium': [serum_sodium],
    'sex': [1 if sex == "Male" else 0],
    'smoking': [1 if smoking == "Yes" else 0]
})

# ── Prediction ────────────────────────────────────────────────────────────────
prob = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mortality Probability", f"{prob*100:.1f}%")

with col2:
    risk_label = "🔴 High Risk" if prob > 0.5 else "🟡 Moderate Risk" if prob > 0.3 else "🟢 Low Risk"
    st.metric("Risk Category", risk_label)

with col3:
    st.metric("Model AUC", "0.805")

# ── Risk gauge ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Mortality Risk Assessment")

fig, ax = plt.subplots(figsize=(10, 1.5))
ax.barh(["Risk"], [prob], color='#F44336' if prob > 0.5 else '#FF9800' if prob > 0.3 else '#4CAF50',
        height=0.5)
ax.barh(["Risk"], [1 - prob], left=[prob], color='#E0E0E0', height=0.5)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax.set_title(f"Predicted Mortality Probability: {prob*100:.1f}%", fontsize=12)
ax.axis('off')
ax.set_xlabel('Mortality Probability')
ax.axis('on')
ax.set_yticks([])
st.pyplot(fig)
plt.close()

# ── Clinical interpretation ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Clinical Interpretation")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Key Risk Indicators:**")
    if ejection_fraction < 30:
        st.error(f"⚠️ Severely reduced ejection fraction ({ejection_fraction}%) — highest risk category")
    elif ejection_fraction < 40:
        st.warning(f"⚠️ Reduced ejection fraction ({ejection_fraction}%) — elevated risk")
    else:
        st.success(f"✅ Ejection fraction ({ejection_fraction}%) — within acceptable range")

    if serum_creatinine > 2.0:
        st.error(f"⚠️ Elevated serum creatinine ({serum_creatinine} mg/dL) — renal impairment")
    elif serum_creatinine > 1.5:
        st.warning(f"⚠️ Borderline serum creatinine ({serum_creatinine} mg/dL)")
    else:
        st.success(f"✅ Serum creatinine ({serum_creatinine} mg/dL) — normal range")

    if serum_sodium < 135:
        st.error(f"⚠️ Low serum sodium ({serum_sodium} mEq/L) — hyponatremia")
    else:
        st.success(f"✅ Serum sodium ({serum_sodium} mEq/L) — normal range")

with col_b:
    st.markdown("**Patient Summary:**")
    st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

# ── SHAP explanation ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Why This Prediction? — SHAP Explanation")

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(input_data)

fig2, ax2 = plt.subplots(figsize=(10, 4))
# handle both 2D and 3D SHAP output structures
if isinstance(shap_vals, list):
    sv = shap_vals[1][0]
else:
    sv = shap_vals[0][:, 1] if shap_vals[0].ndim == 2 else shap_vals[0]
shap_series = pd.Series(sv, index=features).sort_values()
colors = ['#F44336' if v > 0 else '#4CAF50' for v in shap_series]
shap_series.plot(kind='barh', color=colors, ax=ax2)
ax2.axvline(x=0, color='black', linewidth=0.8)
ax2.set_xlabel('SHAP Value (impact on mortality prediction)')
ax2.set_title('Feature Contribution to This Patient\'s Prediction\n(Red = increases mortality risk | Green = decreases risk)')
st.pyplot(fig2)
plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
**Model:** Random Forest (n=200, tuned) | **AUC:** 0.805 | **Dataset:** Heart Failure Clinical Records (299 patients)  
**Author:** Afriyie Karikari Bempah, PharmD | Part of a 10-project healthcare data science portfolio
""")
