# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:21:37 2025

@author: youni
"""

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb
import numpy as np
from tensorflow.keras.models import load_model
import base64
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(page_title="Vu Estimator", layout="wide", page_icon="🧱")

# --- Custom CSS for Styling ---
st.markdown(r"""
<style>
    /* Base container and max width for a clean 100% zoom experience */
    .block-container {
        padding-top: 2rem;
        max-width: 1400px;     /* keep content from stretching too wide */
        margin: 0 auto;        /* center the app on all screens */
    }

    /* Inputs/select width — wide enough to avoid wrapping, not too wide to overflow */
    .stNumberInput > div > div, .stSelectbox > div > div {
        max-width: 260px !important;   /* was 240px */
        min-width: 220px !important;   /* prevent squishing */
    }

    /* Headings & labels */
    .stNumberInput label, .stSelectbox label {
        font-size: 28px !important;
        font-weight: 800;
        white-space: nowrap;  /* keep nice one-line labels */
    }
    .section-header {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 0.8rem;
        white-space: nowrap;
    }

    /* Big banner */
    .form-banner {         
        text-align: center;
        background: linear-gradient(to right, #6a11cb, #2575fc);
        padding: 0.6rem;
        font-size: 40px;
        font-weight: 800;
        color: white;
        border-radius: 10px;
        margin: 1rem 0;
    }

    /* Prediction pill */
    .prediction-result {
        font-size: 20px;
        font-weight: bold;
        color: #2e86ab;
        background-color: #f1f3f4;
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
        margin-top: 1rem;
        white-space: nowrap;
    }

    /* Recent items */
    .recent-box {
        background-color: #f8f9fa;
        padding: 0.6rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Buttons — ensure no letter-per-line wrapping */
    div.stButton > button {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        border: none;
        white-space: nowrap;            /* keep text on one line */
        display: inline-flex;
        align-items: center;
        justify-content: center;
        letter-spacing: normal;
        word-spacing: normal;
        word-break: keep-all;
    }
    div.stButton > button:hover {
        background-color: #27ae60;
    }
    div.stButton:nth-of-type(3) > button {
        background-color: #f28b82 !important;
        color: white !important;
        font-weight: bold !important;
    }
    div.stButton:nth-of-type(3) > button:hover {
        background-color: #e06666 !important;
    }

    /* Keep selectbox control tidy and not cramped */
    div[data-testid="stSelectbox"] > div > div {
        min-width: 240px !important;
    }

    /* Title sizing & logo scaling at 100% zoom */
    h1, .stMarkdown h1, .stHeading h1 {
        line-height: 1.1 !important;
        margin-bottom: .6rem !important;
    }

    /* Right figure — make image responsive inside its column */
    .right-figure img {
        max-width: 100% !important;
        height: auto !important;
        display: block;
        margin: 0 auto;
    }

    /* --- RESPONSIVE LAYOUT FIXES — CSS only (no logic changes) --- */
    @media (max-width: 1600px) {
        .form-banner { font-size: 34px; }
        .section-header { font-size: 24px; }
    }
    @media (max-width: 1400px) {
        h1, .stMarkdown h1, .stHeading h1 { font-size: 36px !important; }
        .stNumberInput > div > div, .stSelectbox > div > div { max-width: 240px !important; }
    }
    @media (max-width: 1200px) {
        h1, .stMarkdown h1, .stHeading h1 { font-size: 32px !important; }
        .form-banner { font-size: 30px; }
        .stNumberInput > div > div, .stSelectbox > div > div { max-width: 220px !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- Logo Display --- 
logo_path = Path("logo2-01.png")
if logo_path.exists():
    with open(logo_path, "rb") as f:
        base64_logo = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 50px;'>
            <img src='data:image/png;base64,{base64_logo}' width='650'>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Title and Info ---
st.title("Ultimate Shear Capacity Estimator for RC Coupling Beams")
st.markdown("This online app predicts the **ultimate shear capacity ($V_u$)** of RC coupling beams by providing only the relevant key input parameters. Powered by machine learning, it delivers **robust and accurate results** for structural design and analysis.")

# --- Load Models and Scalers ---
ann_ps_model = load_model("ANN_PS_Model.keras")
ann_ps_scaler_X = joblib.load("ANN_PS_Scaler_X.save")
ann_ps_scaler_y = joblib.load("ANN_PS_Scaler_y.save")

ann_mlp_model = load_model("ANN_MLP_Model.keras")
ann_mlp_scaler_X = joblib.load("ANN_MLP_Scaler_X.save")
ann_mlp_scaler_y = joblib.load("ANN_MLP_Scaler_y.save")

rf_model = joblib.load("Best_RF_Model.json")

def normalize_input(x_raw, scaler):
    return scaler.transform(x_raw)

def denormalize_output(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled.reshape(-1, 1))[0][0]

@st.cache_resource
def load_models():
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("Best_XGBoost_Model.json")

    cat_model = catboost.CatBoostRegressor()
    cat_model.load_model("Best_CatBoost_Model.cbm")

    lgb_model = lgb.Booster(model_file="Best_LightGBM_Model.txt")

    return {
        "XGBoost": xgb_model,
        "CatBoost": cat_model,
        "LightGBM": lgb_model,
        "PS": ann_ps_model,
        "MLP": ann_mlp_model,
        "Random Forest": rf_model
    }

models = load_models()

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# --- Layout with Two Columns ---
left, right = st.columns([2.2, 1.5], gap="large")

with left:
    st.markdown("<div class='form-banner'>Enter the Design Features for Your Beam</div>", unsafe_allow_html=True)
    st.session_state.input_error = False

    c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='section-header'>Geometric Properties</div>", unsafe_allow_html=True)
    L = st.number_input("Beam Length $l$ (mm)", value=1000.0, min_value=424.0, max_value=2235.0, step=1.0)
    h = st.number_input("Beam Height $h$ (mm)", value=400.0, min_value=169.0, max_value=880.0, step=1.0)
    b = st.number_input("Beam Width $b$ (mm)", value=200.0, min_value=100.0, max_value=406.0, step=1.0)
    AR = st.number_input("Aspect Ratio $l/h$", value=2.5, min_value=0.75, max_value=4.9, step=0.01)

with c2:
    st.markdown("<div class='section-header'>Material Properties</div>", unsafe_allow_html=True)
    fc = st.number_input("Concrete Strength $f'_c$ (MPa)", value=54.0, min_value=18.1, max_value=86.0, step=0.1)
    fyl = st.number_input("Yield Strength of Longitudinal Bars $f_{yl}$ (MPa)", value=476.0, min_value=281.0, max_value=827.0, step=1.0)
    fyv = st.number_input("Yield Strength of Stirrups $f_{yv}$ (MPa)", value=331.0, min_value=212.0, max_value=953.0, step=1.0)
    fyd = st.number_input("Yield Strength of Diagonal Bars $f_{yd}$ (MPa)", value=476.0, min_value=0.0, max_value=883.0, step=1.0)

with c3:
    st.markdown("<div class='section-header'>Reinforcement Details</div>", unsafe_allow_html=True)
    Pl = st.number_input("Longitudinal Reinforcement $\\rho_l$ (%)", value=0.25, min_value=0.09, max_value=4.1, step=0.01)
    Pv = st.number_input("Stirrups Reinforcement $\\rho_v$ (%)", value=0.21, min_value=0.096, max_value=2.9, step=0.001)
    s = st.number_input("Stirrup Spacing $s$ (mm)", value=150.0, min_value=25.0, max_value=500.0, step=1.0)
    Pd = st.number_input("Diagonal Reinforcement $\\rho_d$ (%)", value=1.005, min_value=0.0, max_value=5.8, step=0.01)
    alpha = st.number_input("Diagonal Angle $\\alpha$", value=17.5, min_value=0.0, max_value=45.0, step=1.0)

with right:
    # NOTE: added class 'right-figure' only for CSS responsiveness (no logic change)
    st.markdown("<div class='right-figure'>", unsafe_allow_html=True)
    st.image("beam-01.svg", width=600)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='text-align:center; font-weight:800; font-size:18px;'>RC Coupling Beam Configurations</div>", unsafe_allow_html=True)

    model_choice = st.selectbox("Model Selection", list(models.keys()))

    c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1.2, 1.2])
    with c_btn1:
        submit = st.button("Calculate")
    with c_btn2:
        if st.button("Reset"):
            st.rerun()
    with c_btn3:
        if st.button("Clear All", key="clear_button"):
            st.session_state.results_df = pd.DataFrame()
            st.success("All predictions cleared.")

    if submit and not st.session_state.input_error:
        input_array = np.array([[L, h, b, AR, fc, fyl, fyv, Pl, Pv, s, Pd, fyd, alpha]])
        input_df = pd.DataFrame(input_array, columns=['L','h','b','AR','f′c','fyl','fyv','Pl','Pv','s','Pd','fyd','α֯'])
        model = models[model_choice]

        if model_choice == "LightGBM":
            pred = model.predict(input_df)[0]
        elif model_choice == "PS":
            input_norm = normalize_input(input_array, ann_ps_scaler_X)
            pred_scaled = model.predict(input_norm)[0][0]
            pred = denormalize_output(pred_scaled, ann_ps_scaler_y)
        elif model_choice == "MLP":
            input_norm = normalize_input(input_array, ann_mlp_scaler_X)
            pred_scaled = model.predict(input_norm)[0][0]
            pred = denormalize_output(pred_scaled, ann_mlp_scaler_y)
        else:
            pred = model.predict(input_df)[0]

        input_df["Predicted_V (kN)"] = pred
        st.session_state.results_df = pd.concat([st.session_state.results_df, input_df], ignore_index=True)
        st.markdown(f"<div class='prediction-result'>Predicted Shear Capacity : {pred:.2f} kN</div>", unsafe_allow_html=True)

    if not st.session_state.results_df.empty:
        st.markdown("### 🧾 Recent Predictions")
        for i, row in st.session_state.results_df.tail(5).reset_index(drop=True).iterrows():
            st.markdown(f"<div class='recent-box'>RC Beam {i+1} ➔ {row['Predicted_V (kN)']:.2f} kN</div>", unsafe_allow_html=True)

        csv = st.session_state.results_df.to_csv(index=False)
        st.download_button("📂 Download All Results as CSV", data=csv, file_name="shear_predictions.csv", mime="text/csv", use_container_width=True)

# --- Footer ---
st.markdown("""
<hr style='margin-top: 2rem;'>
<div style='text-align: center; color: #888; font-size: 14px;'>
    Developed by [Your Name]. For academic and research purposes only.
</div>
""", unsafe_allow_html=True)
