# Patient Survival Prediction — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green) ![SHAP](https://img.shields.io/badge/SHAP-explainability-purple) ![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red)

**End-to-end ML pipeline predicting heart failure patient mortality — 3-model comparison, GridSearchCV tuning, SHAP explainability, and deployed Streamlit web app.**

> Author: Afriyie Karikari Bempah, PharmD | [LinkedIn](https://linkedin.com/in/afriyiekarikaribempah) | [GitHub](https://github.com/akbempah1)

---

## 🚀 Live Demo

**[Launch Streamlit App →](https://your-app-url-here.streamlit.app)**

Enter patient clinical parameters and get real-time mortality risk prediction with per-patient SHAP explanation.

---

## Overview

This is the capstone project of a 10-project healthcare data science portfolio. It builds a complete production-ready ML pipeline — from raw clinical data to a deployed web application — using a heart failure dataset of 299 patients.

The Streamlit app allows clinicians and researchers to input patient values via sliders and instantly receive a mortality probability, risk tier, clinical flags, and an explainable SHAP breakdown of which factors drove the prediction.

---

## Key Findings

| Finding | Implication |
|---|---|
| **EF < 30% → 54.8% mortality** | Severely reduced ejection fraction is the most dangerous clinical state |
| **Serum creatinine is #1 SHAP feature** | Kidney failure is the strongest individual mortality predictor |
| **Tuned RF AUC = 0.805** | Strong performance on a small 299-patient dataset |
| **Time variable excluded** | Including follow-up duration would constitute data leakage |
| **Per-patient SHAP explanations** | Every prediction is explainable — critical for clinical trust |

---

## Model Performance

| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic Regression | 0.805 ± 0.044 | 0.729 |
| Random Forest | 0.770 ± 0.041 | 0.793 |
| XGBoost | 0.744 ± 0.027 | 0.738 |
| **Tuned Random Forest** | — | **0.805** |

---

## Streamlit App Features

- **Real-time prediction** — sliders update mortality probability instantly
- **Risk gauge** — visual probability bar with color coding
- **Clinical flags** — automatic warnings for dangerous EF, creatinine, sodium values
- **Per-patient SHAP** — explains which features drove THIS patient's prediction
- **Patient summary table** — all inputs displayed for documentation

---

## Pipeline Components

```
heart_failure_clinical_records_dataset.csv
    ↓
patient_survival_pipeline.ipynb  (EDA → modeling → evaluation → SHAP)
    ↓
survival_model.pkl + features.pkl  (serialized model)
    ↓
app.py  (Streamlit web application)
    ↓
Streamlit Cloud deployment  (public URL)
```

---

## ML Concepts Demonstrated

- Multi-model comparison with stratified cross-validation
- GridSearchCV hyperparameter tuning
- SHAP TreeExplainer (global + per-patient)
- Model serialization with joblib
- Streamlit app development and deployment
- Data leakage identification and prevention

---

## How to Run Locally

```bash
git clone https://github.com/akbempah1/patient-survival-prediction.git
pip install -r requirements.txt
jupyter notebook patient_survival_pipeline.ipynb  # run pipeline first
streamlit run app.py  # launch web app
```

---

## Complete Portfolio

| # | Project | Domain |
|---|---|---|
| 1 | [Medicare Drug Spending](https://github.com/akbempah1/medicare-drug-spending-analysis) | Healthcare |
| 2 | [FDA Adverse Events](https://github.com/akbempah1/fda-adverse-events-analysis) | Healthcare |
| 3 | [Hospital Readmissions](https://github.com/akbempah1/hospital-readmissions-prediction) | ML |
| 4 | [Pharma Portfolio](https://github.com/akbempah1/pharma-stock-portfolio-analysis) | Finance |
| 5 | [Credit Risk Scoring](https://github.com/akbempah1/credit-risk-scoring) | ML/Finance |
| 6 | [Africa Disease Burden](https://github.com/akbempah1/africa-disease-burden-analysis) | Global Health |
| 7 | [Malaria Prediction](https://github.com/akbempah1/malaria-prediction-africa) | ML/Africa |
| 8 | [Pharmacy Forecasting](https://github.com/akbempah1/pharmacy-sales-forecasting) | Time Series |
| 9 | [NCD Risk Factors](https://github.com/akbempah1/ncd-risk-factor-analysis) | Health Equity |
| **10** | **Patient Survival ← you are here** | **End-to-End ML** |

---

*Dataset: Heart Failure Clinical Records via Kaggle. For educational purposes only.*
