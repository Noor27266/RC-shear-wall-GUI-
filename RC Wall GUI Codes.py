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
    .block-container { padding-top: 2rem; }
    .stNumberInput > div > div, .stSelectbox > div > div {
        max-width: 240px !important;
    }
    .stNumberInput label, .stSelectbox label {
        font-size: 28px !important;
        font-weight: 800;
    }
    .section-header {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
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
    .prediction-result {
        font-size: 20px;
        font-weight: bold;
        color: #2e86ab;
        background-color: #f1f3f4;
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
        margin-top: 1rem;
    }
    .recent-box {
        background-color: #f8f9fa;
        padding: 0.6rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 4px solid #4CAF50;
        font-weight: 600;
    }
    div.stButton > button {
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        border: none;
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
</style>
""", unsafe_allow_html=True)

# --- Logo Display --- 
logo_path = Path("logo2-01.png")
if logo_path.exists():
    with open(logo_path, "rb") as f:
        base64_logo = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 50px;'>  <!-- Added margin-top -->
            <img src='data:image/png;base64,{base64_logo}' width='650'>
        </div>
        """,
        unsafe_allow_html=True
    )


# --- Title and Info ---
st.title("Predict Damage index (DI) for RC Shear Walls")
st.markdown("This online app predicts the **Damage Index (DI)** of RC Shear Walls by providing only the relevant key input parameters. Powered by machine learning, it delivers **robust and accurate results** for structural design and analysis.")

# --- Load Models and Scalers ---
# [YOUR ORIGINAL MODEL LOADING CODE HERE - UNCHANGED]
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

# --- Input Parameters for RC Shear Walls ---
R = {
    "lw":(400.0,3500.0), "hw":(495.0,5486.4), "tw":(26.0,305.0), "fc":(13.38,93.6),
    "fyt":(0.0,1187.0), "fysh":(0.0,1375.0), "fyl":(160.0,1000.0), "fybl":(0.0,900.0),
    "rt":(0.000545,0.025139), "rsh":(0.0,0.041888), "rl":(0.0,0.029089), "rbl":(0.0,0.031438),
    "axial":(0.0,0.86), "b0":(45.0,3045.0), "db":(0.0,500.0), "s_db":(0.0,47.65625),
    "AR":(0.388889,5.833333), "M_Vlw":(0.388889,4.1), "theta":(0.0275,4.85),
}

def dv(R, key, proposed): 
    lo, hi = R[key]
    return float(max(lo, min(proposed, hi)))

U = lambda s: rf"\;(\mathrm{{{s}}})"

GEOM = [
    (rf"$l_w{U('mm')}$","lw",1000.0,1.0,None,"Length"),
    (rf"$h_w{U('mm')}$","hw",495.0,1.0,None,"Height"),
    (rf"$t_w{U('mm')}$","tw",200.0,1.0,None,"Thickness"),
    (rf"$b_0{U('mm')}$","b0",200.0,1.0,None,"Boundary element width"),
    (rf"$d_b{U('mm')}$","db",400.0,1.0,None,"Boundary element length"),
    (r"$AR$","AR",2.0,0.01,None,"Aspect ratio"),
    (r"$M/(V_{l_w})$","M_Vlw",2.0,0.01,None,"Shear span ratio"),
]

MATS = [
    (rf"$f'_c{U('MPa')}$",        "fc",   40.0, 0.1, None, "Concrete strength"),
    (rf"$f_{{yt}}{U('MPa')}$",    "fyt",  400.0, 1.0, None, "Transverse web yield strength"),
    (rf"$f_{{ysh}}{U('MPa')}$",   "fysh", 400.0, 1.0, None, "Transverse boundary yield strength"),
    (rf"$f_{{yl}}{U('MPa')}$","fyl",  400.0, 1.0, None, "Vertical web yield strength"),
    (rf"$f_{{ybl}}{U('MPa')}$","fybl", 400.0, 1.0, None, "Vertical boundary yield strength"),
]

REINF = [
    (r"$\rho_t\;(\%)$","rt",0.25,0.0001,"%.6f","Transverse web ratio"),
    (r"$\rho_{sh}\;(\%)$","rsh",0.25,0.0001,"%.6f","Transverse boundary ratio"),
    (r"$\rho_l\;(\%)$","rl",0.25,0.0001,"%.6f","Vertical web ratio"),
    (r"$\rho_{bl}\;(\%)$","rbl",0.25,0.0001,"%.6f","Vertical boundary ratio"),
    (r"$s/d_b$","s_db",0.25,0.01,None,"Hoop spacing ratio"),
    (r"$P/(A_g f'_c)$","axial",0.10,0.001,None,"Axial Load Ratio"),
    (r"$\theta\;(\%)$","theta",4.85,0.0005,None,"Drift Ratio"),
]

def num(label, key, default, step, fmt, help_):
    return st.number_input(
        label, value=dv(R, key, default), step=step,
        min_value=R[key][0], max_value=R[key][1],
        format=fmt if fmt else None, help=help_
    )

# --- Layout with Two Columns ---
left, right = st.columns([2.2, 1.5], gap="large")

with left:
    st.markdown("<div class='form-banner'>Inputs Features</div>", unsafe_allow_html=True)
    st.session_state.input_error = False

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("<div class='section-header'>Geometry</div>", unsafe_allow_html=True)
        lw, hw, tw, b0, db, AR, M_Vlw = [num(*row) for row in GEOM]

    with c2:
        st.markdown("<div class='section-header'>Reinf. Ratios</div>", unsafe_allow_html=True)
        rt, rsh, rl, rbl, s_db, axial, theta = [num(*row) for row in REINF]

    with c3:
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fc, fyt, fysh = [num(*row) for row in MATS[:3]]
        fyl, fybl = [num(*row) for row in MATS[3:]]

with right:
    # Display second logo/image
    try:
        second_logo_path = Path("TJU logo.png")
        if second_logo_path.exists():
            with open(second_logo_path, "rb") as f:
                base64_second_logo = base64.b64encode(f.read()).decode()
            st.markdown(
                f"""
                <div style='text-align: right; margin-top: 20px;'>
                    <img src='data:image/png;base64,{base64_second_logo}' width='200'>
                </div>
                """,
                unsafe_allow_html=True
            )
    except:
        pass
    
    # DI vs Theta Chart Placeholder
    st.markdown("<div class='section-header'>DI vs θ Curve</div>", unsafe_allow_html=True)
    chart_placeholder = st.empty()
    
    # Display a simple chart (you can replace this with your actual chart)
    try:
        import altair as alt
        # Sample data for demonstration
        theta_values = np.linspace(0, 4.85, 50)
        di_values = np.minimum(1.5, np.maximum(0.035, theta_values * 0.3))
        chart_data = pd.DataFrame({'θ': theta_values, 'DI': di_values})
        
        chart = alt.Chart(chart_data).mark_line().encode(
            x='θ:Q',
            y='DI:Q'
        ).properties(
            width=400,
            height=300
        )
        chart_placeholder.altair_chart(chart, use_container_width=True)
    except:
        st.info("Chart will be displayed here")

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

    # Prediction logic for RC Shear Walls
    if submit and not st.session_state.input_error:
        # Create input array for shear walls
        input_array = np.array([[lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta]])
        input_df = pd.DataFrame(input_array, columns=[
            'l_w','h_w','t_w','f′c','fyt','fysh','fyl','fybl','ρt','ρsh','ρl','ρbl',
            'P/(Agf′c)','b0','db','s/db','AR','M/Vlw','θ'
        ])
        
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

        # Ensure prediction is within reasonable bounds for DI
        pred = max(0.035, min(pred, 1.5))
        
        input_df["Predicted_DI"] = pred
        st.session_state.results_df = pd.concat([st.session_state.results_df, input_df], ignore_index=True)
        st.markdown(f"<div class='prediction-result'>Predicted Damage Index (DI): {pred:.4f}</div>", unsafe_allow_html=True)

    if not st.session_state.results_df.empty:
        st.markdown("### 🧾 Recent Predictions")
        for i, row in st.session_state.results_df.tail(5).reset_index(drop=True).iterrows():
            st.markdown(f"<div class='recent-box'>Pred {i+1} ➔ DI = {row['Predicted_DI']:.4f}</div>", unsafe_allow_html=True)

        csv = st.session_state.results_df.to_csv(index=False)
        st.download_button("📂 Download as CSV", data=csv, file_name="di_predictions.csv", mime="text/csv", use_container_width=True)

# --- Footer ---
st.markdown("""
<hr style='margin-top: 2rem;'>
<div style='text-align: center; color: #888; font-size: 14px;'>
    Developed for RC Shear Wall Damage Index Prediction. For academic and research purposes only.
</div>
""", unsafe_allow_html=True)
