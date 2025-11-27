DOC_NOTES = """
RC Shear Wall Damage Index (DI) Estimator — compact, same logic/UI
"""

# =============================================================================
# 🚀 STEP 1: CORE IMPORTS & SETUP
# =============================================================================
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import streamlit as st
import pandas as pd
import numpy as np
import base64, json
from pathlib import Path
from glob import glob
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb

# Keras compatibility loader
try:
    from tensorflow.keras.models import load_model as _tf_load_model
except Exception:
    _tf_load_model = None
try:
    from keras.models import load_model as _k3_load_model
except Exception:
    _k3_load_model = None

def _load_keras_model(path):
    errs = []
    if _tf_load_model is not None:
        try: return _tf_load_model(path)
        except Exception as e: errs.append(f"tf.keras: {e}")
    if _k3_load_model is not None:
        try: return _k3_load_model(path)
        except Exception as e: errs.append(f"keras: {e}")
    raise RuntimeError(" / ".join(errs) if errs else "No Keras loader available")

# Session state
st.session_state.setdefault("results_df", pd.DataFrame())

# =============================================================================
# 🎨 STEP 2: UI CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

# CSS Styling
SCALE_UI = 0.36
s = lambda v: int(round(v * SCALE_UI))
FS_TITLE, FS_SECTION, FS_LABEL, FS_UNITS = s(20), s(60), s(50), s(30)
FS_INPUT, FS_SELECT, FS_BUTTON, FS_BADGE, FS_RECENT = s(30), s(35), s(20), s(30), s(20)
INPUT_H = max(32, int(FS_INPUT * 2.0))
PRIMARY, SECONDARY, INPUT_BG, INPUT_BORDER, LEFT_BG = "#8E44AD", "#f9f9f9", "#ffffff", "#e6e9f2", "#e0e4ec"

st.markdown(f"""
<style>
.block-container {{ padding-top: 0.5rem !important; }}
h1 {{ font-size:{FS_TITLE}px !important; margin:0 rem 0 !important; }}
.section-header {{ font-size:{FS_SECTION}px !important; font-weight:700; margin:.35rem 0; }}

.stNumberInput label, .stSelectbox label {{ font-size:{FS_LABEL}px !important; font-weight:700; }}
.stNumberInput label .katex, .stSelectbox label .katex {{ font-size:{FS_LABEL}px !important; line-height:1.2 !important; }}

div[data-testid="stNumberInput"] input[type="number"],
div[data-testid="stNumberInput"] input[type="text"] {{
    font-size:{FS_INPUT}px !important; height:{INPUT_H}px !important;
    line-height:{INPUT_H - 8}px !important; font-weight:600 !important;
    padding:10px 12px !important;
}}

div[data-testid="stNumberInput"] [data-baseweb*="input"] {{
    background:{INPUT_BG} !important; border:1px solid {INPUT_BORDER} !important;
    border-radius:12px !important; box-shadow:0 1px 2px rgba(16,24,40,.06) !important;
}}
div[data-testid="stNumberInput"] [data-baseweb*="input"]:hover {{ border-color:#d6dced !important; }}
div[data-testid="stNumberInput"] [data-baseweb*="input"]:focus-within {{
    border-color:{PRIMARY} !important; box-shadow:0 0 0 3px rgba(106,17,203,.15) !important;
}}

.stSelectbox [role="combobox"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child {{
    font-size:{FS_SELECT}px !important;
}}

div.stButton > button {{
    font-size:{FS_BUTTON}px !important; height:{max(42, int(round(FS_BUTTON*1.45)))}px !important;
    line-height:{max(36, int(round(FS_BUTTON*1.15)))}px !important; white-space:nowrap !important;
    color:#fff !important; font-weight:700; border:none !important; border-radius:8px !important;
    background:#4CAF50 !important;
}}

.prediction-result {{
    font-size:{FS_BADGE}px !important; font-weight:700; color:#2e86ab;
    background:#f1f3f4; padding:.6rem; border-radius:6px; text-align:center; margin-top:.6rem;
}}

/* Full page background */
html, body, #root, .stApp, section.main, .block-container, [data-testid="stAppViewContainer"] {{
    background: linear-gradient(90deg, #e0e4ec 60%, transparent 60%) !important;
    min-height: 100vh !important; height: auto !important;
}}

[data-testid="column"]:first-child {{ min-height: 100vh !important; background: #e0e4ec !important; }}

/* Hide number input buttons */
div[data-testid="stNumberInput"] button {{ display: none !important; }}

/* Form banner */
.form-banner{{
  background: linear-gradient(90deg, #0E9F6E, #84CC16) !important;
  color: #fff !important; text-align: center !important;
  border-radius: 10px !important; padding: .45rem .75rem !important;
  margin-top: 65px !important; transform: translateY(0) !important;
}}

/* Chart positioning */
#di_theta_chart_wrapper {{ margin-top:-360px !important; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🤖 STEP 3: MODEL LOADING
# =============================================================================
class _ScalerShim:
    def __init__(self, X_scaler, Y_scaler):
        import numpy as _np
        self._np = _np
        self.Xs = X_scaler
        self.Ys = Y_scaler

    def transform_X(self, X): return self.Xs.transform(X)
    def inverse_transform_y(self, y):
        y = self._np.array(y).reshape(-1, 1)
        return self.Ys.inverse_transform(y)

def pfind(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists(): return p
    roots = [BASE_DIR, Path.cwd(), Path("/mnt/data")]
    for root in roots:
        if not root.exists(): continue
        for c in candidates:
            p = root / c
            if p.exists(): return p
    for root in [BASE_DIR, Path("/mnt/data")]:
        if not root.exists(): continue
        for sub in root.iterdir():
            if sub.is_dir():
                for c in candidates:
                    p = sub / c
                    if p.exists(): return p
    pats = []
    for c in candidates:
        for root in [BASE_DIR, Path.cwd(), Path("/mnt/data")]:
            if root.exists(): pats.append(str(root / "**" / c))
    for pat in pats:
        matches = glob(pat, recursive=True)
        if matches: return Path(matches[0])
    raise FileNotFoundError(f"None of these files were found: {candidates}")

BASE_DIR = Path(__file__).resolve().parent
health = []
model_registry = {}

# Load PS (ANN)
ann_ps_model, ann_ps_proc = None, None
try:
    ps_model_path = pfind(["ANN_PS_Model.keras", "ANN_PS_Model.h5"])
    ann_ps_model = _load_keras_model(ps_model_path)
    sx = joblib.load(pfind(["ANN_PS_Scaler_X.save", "ANN_PS_Scaler_X.pkl", "ANN_PS_Scaler_X.joblib"]))
    sy = joblib.load(pfind(["ANN_PS_Scaler_y.save", "ANN_PS_Scaler_y.pkl", "ANN_PS_Scaler_y.joblib"]))
    ann_ps_proc = _ScalerShim(sx, sy)
    model_registry["PS"] = ann_ps_model
    health.append(("PS (ANN)", True, f"loaded from {ps_model_path}"))
except Exception as e: health.append(("PS (ANN)", False, f"{e}"))

# Load MLP (ANN)
ann_mlp_model, ann_mlp_proc = None, None
try:
    mlp_model_path = pfind(["ANN_MLP_Model.keras", "ANN_MLP_Model.h5"])
    ann_mlp_model = _load_keras_model(mlp_model_path)
    sx = joblib.load(pfind(["ANN_MLP_Scaler_X.save", "ANN_MLP_Scaler_X.pkl", "ANN_MLP_Scaler_X.joblib"]))
    sy = joblib.load(pfind(["ANN_MLP_Scaler_y.save", "ANN_MLP_Scaler_y.pkl", "ANN_MLP_Scaler_y.joblib"]))
    ann_mlp_proc = _ScalerShim(sx, sy)
    model_registry["MLP"] = ann_mlp_model
    health.append(("MLP (ANN)", True, f"loaded from {mlp_model_path}"))
except Exception as e: health.append(("MLP (ANN)", False, f"{e}"))

# Load Random Forest
rf_model = None
try:
    rf_path = pfind(["random_forest_model.pkl", "random_forest_model.joblib", "rf_model.pkl", "RF_model.pkl"])
    rf_model = joblib.load(rf_path)
    model_registry["Random Forest"] = rf_model
    health.append(("Random Forest", True, f"loaded from {rf_path}"))
except Exception as e: health.append(("Random Forest", False, str(e)))

# Load XGBoost
xgb_model = None
try:
    xgb_path = pfind(["XGBoost_trained_model_for_DI.json", "Best_XGBoost_Model.json", "xgboost_model.json"])
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    model_registry["XGBoost"] = xgb_model
    health.append(("XGBoost", True, f"loaded from {xgb_path}"))
except Exception as e: health.append(("XGBoost", False, str(e)))

# Load CatBoost
cat_model = None
try:
    cat_path = pfind(["CatBoost.cbm", "Best_CatBoost_Model.cbm", "catboost.cbm"])
    cat_model = catboost.CatBoostRegressor()
    cat_model.load_model(cat_path)
    model_registry["CatBoost"] = cat_model
    health.append(("CatBoost", True, f"loaded from {cat_path}"))
except Exception as e: health.append(("CatBoost", False, f"{e}"))

# Load LightGBM
lgb_model = None
try:
    p = pfind(["LightGBM_model.txt", "Best_LightGBM_Model.txt", "LightGBM_model.bin"])
    lgb_model = lgb.Booster(model_file=str(p))
    model_registry["LightGBM"] = lgb_model
    health.append(("LightGBM", True, f"loaded from {p}"))
except Exception as e: health.append(("LightGBM", False, str(e)))

# =============================================================================
# 📊 STEP 4: INPUT PARAMETERS & LAYOUT
# =============================================================================
R = {
    "lw": (400.0, 3500.0), "hw": (495.0, 5486.4), "tw": (26.0, 305.0), "fc": (13.38, 93.6),
    "fyt": (0.0, 1187.0), "fysh": (0.0, 1375.0), "fyl": (160.0, 1000.0), "fybl": (0.0, 900.0),
    "rt": (0.000545, 0.025139), "rsh": (0.0, 0.041888), "rl": (0.0, 0.029089), "rbl": (0.0, 0.031438),
    "axial": (0.0, 0.86), "b0": (45.0, 3045.0), "db": (0.0, 500.0), "s_db": (0.0, 47.65625),
    "AR": (0.388889, 5.833333), "M_Vlw": (0.388889, 4.1), "theta": (0.0275, 4.85),
}
THETA_MAX = R["theta"][1]

def dv(R, key, proposed):
    lo, hi = R[key]
    return float(max(lo, min(proposed, hi)))

def num(label, key, default, step, fmt, help_):
    return st.number_input(label, value=dv(R, key, default), step=step, 
                          min_value=R[key][0], max_value=R[key][1], 
                          format=fmt if fmt else None, help=help_)

U = lambda s: rf"\;(\mathrm{{{s}}})"

GEOM = [
    (rf"$l_w{U('mm')}$", "lw", 1000.0, 1.0, None, "Length"),
    (rf"$h_w{U('mm')}$", "hw", 495.0, 1.0, None, "Height"),
    (rf"$t_w{U('mm')}$", "tw", 200.0, 1.0, None, "Thickness"),
    (rf"$b_0{U('mm')}$", "b0", 200.0, 1.0, None, "Boundary element width"),
    (rf"$d_b{U('mm')}$", "db", 400.0, 1.0, None, "Boundary element length"),
    (r"$AR$", "AR", 2.0, 0.01, None, "Aspect ratio"),
    (r"$M/(V_{l_w})$", "M_Vlw", 2.0, 0.01, None, "Shear span ratio"),
]

MATS = [
    (rf"$f'_c{U('MPa')}$", "fc", 40.0, 0.1, None, "Concrete strength"),
    (rf"$f_{{yt}}{U('MPa')}$", "fyt", 400.0, 1.0, None, "Transverse web yield strength"),
    (rf"$f_{{ysh}}{U('MPa')}$", "fysh", 400.0, 1.0, None, "Transverse boundary yield strength"),
    (rf"$f_{{yl}}{U('MPa')}$", "fyl", 400.0, 1.0, None, "Vertical web yield strength"),
    (rf"$f_{{ybl}}{U('MPa')}$", "fybl", 400.0, 1.0, None, "Vertical boundary yield strength"),
]

REINF = [
    (r"$\rho_t\;(\%)$", "rt", 0.25, 0.0001, "%.6f", "Transverse web ratio"),
    (r"$\rho_{sh}\;(\%)$", "rsh", 0.25, 0.0001, "%.6f", "Transverse boundary ratio"),
    (r"$\rho_l\;(\%)$", "rl", 0.25, 0.0001, "%.6f", "Vertical web ratio"),
    (r"$\rho_{bl}\;(\%)$", "rbl", 0.25, 0.0001, "%.6f", "Vertical boundary ratio"),
    (r"$s/d_b$", "s_db", 0.25, 0.01, None, "Hoop spacing ratio"),
    (r"$P/(A_g f'_c)$", "axial", 0.10, 0.001, None, "Axial Load Ratio"),
    (r"$\theta\;(\%)$", "theta", THETA_MAX, 0.0005, None, "Drift Ratio"),
]

left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown("""
    <div style="background:transparent; border-radius:12px; padding:0px; margin:-20px 0 0 0; box-shadow:none;">
        <div style="text-align:center; font-size:25px; font-weight:600; color:#333; margin:0; padding:2px;">
            Predict Damage index (DI) for RC Shear Walls
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1px;'></div>" * 3, unsafe_allow_html=True)
    st.markdown("<div class='form-banner'>Inputs Features</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1], gap="small")
    with c1:
        st.markdown("<div class='section-header'>Geometry </div>", unsafe_allow_html=True)
        lw, hw, tw, b0, db, AR, M_Vlw = [num(*row) for row in GEOM]
    with c2:
        st.markdown("<div class='section-header'>Reinf. Ratios </div>", unsafe_allow_html=True)
        rt, rsh, rl, rbl, s_db, axial, theta = [num(*row) for row in REINF]
    with c3:
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fc, fyt, fysh = [num(*row) for row in MATS[:3]]
        fyl, fybl = [num(*row) for row in MATS[3:]]

# =============================================================================
# 🎮 STEP 5: RIGHT PANEL - CONTROLS
# =============================================================================
with right:
    # Model selection and buttons
    available = set(model_registry.keys())
    order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
    ordered_keys = [m for m in order if m in available] or ["(no models loaded)"]
    display_labels = ["RF" if m == "Random Forest" else m for m in ordered_keys]
    _label_to_key = {"RF": "Random Forest"}
    
    model_choice_label = st.selectbox("Model Selection", display_labels, key="model_select_compact")
    model_choice = _label_to_key.get(model_choice_label, model_choice_label)

    submit = st.button("Calculate", key="calc_btn", use_container_width=True)
    if st.button("Reset", key="reset_btn", use_container_width=True): st.rerun()
    if st.button("Clear All", key="clear_btn", use_container_width=True): 
        st.session_state.results_df = pd.DataFrame()

    # Prediction display
    if not st.session_state.results_df.empty:
        latest_pred = st.session_state.results_df.iloc[-1]["Predicted_DI"]
        st.markdown(f"<div class='prediction-result'>Predicted Damage Index (DI): {latest_pred:.4f}</div>", 
                   unsafe_allow_html=True)
        
        csv = st.session_state.results_df.to_csv(index=False)
        st.download_button("📂 Download as CSV", data=csv, file_name="di_predictions.csv", 
                          mime="text/csv", use_container_width=True, key="dl_csv_main")

    # Chart slot
    st.markdown("<div id='di_theta_chart_wrapper' style='margin-top:-360px;'>", unsafe_allow_html=True)
    chart_slot = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 🔮 STEP 6: PREDICTION ENGINE
# =============================================================================
_TRAIN_NAME_MAP = {
    "l_w": "lw", "h_w": "hw", "t_w": "tw", "f′c": "fc", "fyt": "fyt", "fysh": "fysh",
    "fyl": "fyl", "fybl": "fybl", "ρt": "pt", "ρsh": "psh", "ρl": "pl", "ρbl": "pbl",
    "P/(Agf′c)": "P/(Agfc)", "b0": "b0", "db": "db", "s/db": "s/db", "AR": "AR",
    "M/Vlw": "M/Vlw", "θ": "θ",
}
_TRAIN_COL_ORDER = ["lw", "hw", "tw", "fc", "fyt", "fysh", "fyl", "fybl", "pt", "psh", 
                   "pl", "pbl", "P/(Agfc)", "b0", "db", "s/db", "AR", "M/Vlw", "θ"]

def _df_in_train_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_TRAIN_NAME_MAP).reindex(columns=_TRAIN_COL_ORDER)

def predict_di(choice, _unused_array, input_df):
    df_trees = _df_in_train_order(input_df)
    df_trees = df_trees.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = df_trees.values.astype(np.float32)

    if choice == "LightGBM":
        prediction = float(model_registry["LightGBM"].predict(X)[0])
    elif choice == "XGBoost":
        prediction = float(model_registry["XGBoost"].predict(X)[0])
    elif choice == "CatBoost":
        prediction = float(model_registry["CatBoost"].predict(X)[0])
    elif choice == "Random Forest":
        prediction = float(model_registry["Random Forest"].predict(X)[0])
    elif choice == "PS":
        Xn = ann_ps_proc.transform_X(X)
        try: yhat = model_registry["PS"].predict(Xn, verbose=0)[0][0]
        except Exception:
            model_registry["PS"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["PS"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_ps_proc.inverse_transform_y(yhat).item())
    elif choice == "MLP":
        Xn = ann_mlp_proc.transform_X(X)
        try: yhat = model_registry["MLP"].predict(Xn, verbose=0)[0][0]
        except Exception:
            model_registry["MLP"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["MLP"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_mlp_proc.inverse_transform_y(yhat).item())

    return max(0.035, min(prediction, 1.5))

def _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta_val):
    cols = ["l_w", "h_w", "t_w", "f′c", "fyt", "fysh", "fyl", "fybl", "ρt", "ρsh", "ρl", "ρbl", 
            "P/(Agf′c)", "b0", "db", "s/db", "AR", "M/Vlw", "θ"]
    x = np.array([[lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta_val]], 
                dtype=np.float32)
    return pd.DataFrame(x, columns=cols)

def _sweep_curve_df(model_choice, base_df, theta_max=THETA_MAX, step=0.1):
    if model_choice not in model_registry: return pd.DataFrame(columns=["θ", "Predicted_DI"])
    thetas = np.round(np.arange(0.0, theta_max + 1e-9, step), 2)
    rows = []
    for th in thetas:
        df = base_df.copy()
        df.loc[:, "θ"] = float(th)
        di = predict_di(model_choice, None, df)
        di = max(0.035, min(di, 1.5))
        rows.append({"θ": float(th), "Predicted_DI": float(di)})
    return pd.DataFrame(rows)

def render_di_chart(results_df: pd.DataFrame, curve_df: pd.DataFrame, theta_max: float = THETA_MAX, di_max: float = 1.5, size: int = 460):
    import altair as alt
    selection = alt.selection_point(name="select", fields=["θ", "Predicted_DI"], nearest=True, 
                                   on="mouseover", empty=False, clear="mouseout")
    
    AXIS_LABEL_FS, AXIS_TITLE_FS, TICK_SIZE, TITLE_PAD, LABEL_PAD = 14, 16, 6, 10, 6
    base_axes_df = pd.DataFrame({"θ": [0.0, theta_max], "Predicted_DI": [0.0, 0.0]})
    x_ticks = np.linspace(0.0, theta_max, 5).round(2)

    axes_layer = (alt.Chart(base_axes_df).mark_line(opacity=0).encode(
        x=alt.X("θ:Q", title="Drift Ratio (θ)", scale=alt.Scale(domain=[0, theta_max], nice=False, clamp=True),
               axis=alt.Axis(values=list(x_ticks), labelFontSize=AXIS_LABEL_FS, titleFontSize=AXIS_TITLE_FS,
                            labelPadding=LABEL_PAD, titlePadding=TITLE_PAD, tickSize=TICK_SIZE)),
        y=alt.Y("Predicted_DI:Q", title="Damage Index (DI)", scale=alt.Scale(domain=[0, di_max], nice=False, clamp=True),
               axis=alt.Axis(values=[0.0, 0.2, 0.5, 1.0, 1.5], labelFontSize=AXIS_LABEL_FS, titleFontSize=AXIS_TITLE_FS,
                            labelPadding=LABEL_PAD, titlePadding=TITLE_PAD, tickSize=TICK_SIZE)))
        .properties(width=size, height=size))

    curve = curve_df if (curve_df is not None and not curve_df.empty) else pd.DataFrame({"θ": [], "Predicted_DI": []})
    line_layer = alt.Chart(curve).mark_line(strokeWidth=2).encode(x="θ:Q", y="Predicted_DI:Q").properties(width=size, height=size)

    k = 3
    curve_points = curve.iloc[::k].copy() if not curve.empty else pd.DataFrame({"θ": [], "Predicted_DI": []})
    points_layer = alt.Chart(curve_points).mark_circle(size=60, opacity=0.7).encode(
        x="θ:Q", y="Predicted_DI:Q",
        tooltip=[alt.Tooltip("θ:Q", title="Drift Ratio (θ)", format=".2f"),
                alt.Tooltip("Predicted_DI:Q", title="Predicted DI", format=".4f")]).add_params(selection)

    rules_layer = alt.Chart(curve).mark_rule(color="red", strokeWidth=2).encode(x="θ:Q", y="Predicted_DI:Q").transform_filter(selection)
    text_layer = alt.Chart(curve).mark_text(align="left", dx=8, dy=-8, fontSize=14, fontWeight="bold", color="red").encode(
        x="θ:Q", y="Predicted_DI:Q", text=alt.Text("Predicted_DI:Q", format=".4f")).transform_filter(selection)

    chart = alt.layer(axes_layer, line_layer, points_layer, rules_layer, text_layer
                    ).configure_view(strokeWidth=0).configure_axis(domain=True, ticks=True
                    ).configure(padding={"left": 6, "right": 6, "top": 6, "bottom": 6})
    
    chart_html = chart.to_html()
    chart_html = chart_html.replace("</style>", "</style><style>.vega-embed .vega-tooltip, .vega-embed .vega-tooltip * "
    "{ font-size: 14px !important; font-weight: bold !important; background: #000 !important; color: #fff !important; padding: 12px !important; }</style>")
    st.components.v1.html(chart_html, height=size + 100)

# =============================================================================
# ⚡ STEP 7: EXECUTION & VISUALIZATION
# =============================================================================
_order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
_label_to_key = {"RF": "Random Forest"}

def _pick_default_model():
    for m in _order:
        if m in model_registry: return m
    return None

if "model_choice" not in locals():
    _label = st.session_state.get("model_select_compact") or st.session_state.get("model_select")
    model_choice = _label_to_key.get(_label, _label) if _label is not None else _pick_default_model()

if (model_choice is None) or (model_choice not in model_registry):
    st.error("No trained model is available. Please check the Model Selection on the right.")
else:
    if "submit" in locals() and submit:
        xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
        try:
            pred = predict_di(model_choice, None, xdf)
            row = xdf.copy()
            row["Predicted_DI"] = pred
            st.session_state.results_df = pd.concat([st.session_state.results_df, row], ignore_index=True)
        except Exception as e: st.error(f"Prediction failed for {model_choice}: {e}")

    _base_xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
    _curve_df = _sweep_curve_df(model_choice, _base_xdf, theta_max=THETA_MAX, step=0.1)

    with chart_slot:
        render_di_chart(st.session_state.results_df, _curve_df, theta_max=THETA_MAX, di_max=1.5, size=400)
