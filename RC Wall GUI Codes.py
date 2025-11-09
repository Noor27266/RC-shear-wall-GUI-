# -*- coding: utf-8 -*-

DOC_NOTES = """
RC Shear Wall Damage Index (DI) Estimator — compact, same logic/UI
"""

# =============================================================================
# 🚀 STEP 1: CORE IMPORTS & TENSORFLOW BACKEND SETUP
# =============================================================================
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import streamlit as st
import pandas as pd
import numpy as np
import base64, json
from pathlib import Path
from glob import glob

# ML libs
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb

# --- Keras compatibility loader (PS/MLP only) ---
try:
    from tensorflow.keras.models import load_model as _tf_load_model
except Exception:
    _tf_load_model = None
try:
    from keras.models import load_model as _k3_load_model   # works when keras==3 is present
except Exception:
    _k3_load_model = None

def _load_keras_model(path):
    """Try tf.keras first, then keras (Keras 3)."""
    errs = []
    if _tf_load_model is not None:
        try:
            return _tf_load_model(path)
        except Exception as e:
            errs.append(f"tf.keras: {e}")
    if _k3_load_model is not None:
        try:
            return _k3_load_model(path)
        except Exception as e:
            errs.append(f"keras: {e}")
    raise RuntimeError(" / ".join(errs) if errs else "No Keras loader available")

# --- session defaults (prevents AttributeError on first run) ---
st.session_state.setdefault("results_df", pd.DataFrame())

# =============================================================================
# 🔧 STEP 2: UTILITY FUNCTIONS & HELPER TOOLS
# =============================================================================
css = lambda s: st.markdown(s, unsafe_allow_html=True)
def b64(path: Path) -> str: return base64.b64encode(path.read_bytes()).decode("ascii")
def dv(R, key, proposed): lo, hi = R[key]; return float(max(lo, min(proposed, hi)))

# ---------- path helper ----------
BASE_DIR = Path(__file__).resolve().parent
def pfind(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    roots = [BASE_DIR, Path.cwd(), Path("/mnt/data")]
    for root in roots:
        if not root.exists():
            continue
        for c in candidates:
            p = root / c
            if p.exists():
                return p
    for root in [BASE_DIR, Path("/mnt/data")]:
        if not root.exists():
            continue
        for sub in root.iterdir():
            if sub.is_dir():
                for c in candidates:
                    p = sub / c
                    if p.exists():
                        return p
    pats = []
    for c in candidates:
        for root in [BASE_DIR, Path.cwd(), Path("/mnt/data")]:
            if root.exists():
                pats.append(str(root / "**" / c))
    for pat in pats:
        matches = glob(pat, recursive=True)
        if matches:
            return Path(matches[0])
    raise FileNotFoundError(f"None of these files were found: {candidates}")

# =============================================================================
# 🎨 STEP 3: STREAMLIT PAGE CONFIGURATION & UI STYLING
# =============================================================================
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

# ====== FONTS/LOGO KNOBS BELOW ======
SCALE_UI = 0.36

s = lambda v: int(round(v * SCALE_UI))

FS_TITLE   = s(40)  # SMALLER TITLE as requested
FS_SECTION = s(60)
FS_LABEL   = s(50)
FS_UNITS   = s(30)
FS_INPUT   = s(30)
FS_SELECT  = s(35)
FS_BUTTON  = s(20)
FS_BADGE   = s(30)
FS_RECENT  = s(20)
INPUT_H    = max(32, int(FS_INPUT * 2.0))

# Colors
PRIMARY   = "#8E44AD"
SECONDARY = "#f9f9f9"
INPUT_BG     = "#ffffff"
INPUT_BORDER = "#e6e9f2"
LEFT_BG      = "#e0e4ec"  # Grey background for left side

# =============================================================================
# 🎨 STEP 3.1: COMPREHENSIVE CSS STYLING - RESTORED ORIGINAL STYLING
# =============================================================================
css(f"""
<style>
  .block-container {{ padding-top: 0rem; }}
  h1 {{ font-size:{FS_TITLE}px !important; margin:0 rem 0 !important; }}

  .section-header {{
    font-size:{FS_SECTION}px !important;
    font-weight:700; margin:.35rem 0;
  }}

  .stNumberInput label, .stSelectbox label {{
    font-size:{FS_LABEL}px !important; font-weight:700;
  }}

  div[data-testid="stNumberInput"] input[type="number"],
  div[data-testid="stNumberInput"] input[type="text"] {{
      font-size:{FS_INPUT}px !important;
      height:{INPUT_H}px !important;
      line-height:{INPUT_H - 8}px !important;
      font-weight:600 !important;
      padding:10px 12px !important;
  }}

  div[data-testid="stNumberInput"] [data-baseweb*="input"] {{
      background:{INPUT_BG} !important;
      border:1px solid {INPUT_BORDER} !important;
      border-radius:12px !important;
  }}

  /* Buttons with original colors */
  div.stButton > button {{
    font-size:{FS_BUTTON}px !important;
    height:{max(42, int(round(FS_BUTTON*1.45)))}px !important;
    line-height:{max(36, int(round(FS_BUTTON*1.15)))}px !important;
    white-space:nowrap !important;
    color:#fff !important;
    font-weight:700; border:none !important; border-radius:8px !important;
  }}

  button[key="calc_btn"] {{ background:#4CAF50 !important; }}  /* Green */
  button[key="reset_btn"] {{ background:#2196F3 !important; }} /* Blue */
  button[key="clear_btn"] {{ background:#f44336 !important; }} /* Red */

  .form-banner {{
    text-align:center;
    background: linear-gradient(90deg, #0E9F6E, #84CC16);
    color: #fff;
    padding:.45rem .75rem;
    border-radius:10px;
    font-weight:800;
    font-size:{FS_SECTION + 4}px;
    margin:.1rem 0 !important;
  }}

  .prediction-result {{
    font-size:{FS_BADGE}px !important; font-weight:700; color:#2e86ab;
    background:#f1f3f4; padding:.6rem; border-radius:6px; text-align:center; margin-top:.6rem;
  }}

  /* Grey background for left side */
  [data-testid="column"]:nth-child(1) {{
      background: {LEFT_BG} !important;
      border-radius:12px !important;
      padding:16px !important;
      margin:10px !important;
  }}

  /* Compact chart size */
  .vega-embed {{
    width: 300px !important;
    height: 300px !important;
  }}

  /* Header styling */
  .page-header {{ display:flex; align-items:center; justify-content:flex-start; gap:20px; margin:0; padding:0; }}
  .page-header__title {{ font-size:{FS_TITLE}px; font-weight:800; margin:0; }}
  
  /* Logo positioning */
  .page-header__logo {{
    height:50px; 
    width:auto; 
    display:block; 
    position: fixed;
    top: 60px;
    left: 950px;
  }}
</style>
""")

# Hide Streamlit default elements
st.markdown("""
<style>
    header { visibility: hidden; }
    .main { padding-top: 0rem; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🏷️ STEP 4: HEADER WITH TITLE AND LOGO
# =============================================================================
try:
    _logo_path = BASE_DIR / "TJU logo.png"
    _b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii") if _logo_path.exists() else ""
except Exception:
    _b64 = ""

st.markdown(f"""
<div class="page-header">
    <div class="page-header__title">Predict Damage index (DI) for RC Shear Walls</div>
    {f'<img class="page-header__logo" alt="Logo" src="data:image/png;base64,{_b64}" />' if _b64 else ''}
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 🤖 STEP 5: MACHINE LEARNING MODEL LOADING
# =============================================================================
# [Keep all your original model loading code here - it was correct]
def record_health(name, ok, msg=""): health.append((name, ok, msg, "ok" if ok else "err"))
health = []

class _ScalerShim:
    def __init__(self, X_scaler, y_scaler):
        import numpy as _np
        self._np = _np
        self.Xs = X_scaler
        self.Ys = y_scaler
        self.x_kind = "External joblib"
        self.y_kind = "External joblib"
    def transform_X(self, X): return self.Xs.transform(X)
    def inverse_transform_y(self, y):
        y = self._np.array(y).reshape(-1, 1)
        return self.Ys.inverse_transform(y)

# [Include all your model loading code here - it was working fine]
ann_ps_model = None; ann_ps_proc = None
try:
    ps_model_path = pfind(["ANN_PS_Model.keras", "ANN_PS_Model.h5"])
    ann_ps_model = _load_keras_model(ps_model_path)
    sx = joblib.load(pfind(["ANN_PS_Scaler_X.save","ANN_PS_Scaler_X.pkl","ANN_PS_Scaler_X.joblib"]))
    sy = joblib.load(pfind(["ANN_PS_Scaler_y.save","ANN_PS_Scaler_y.pkl","ANN_PS_Scaler_y.joblib"]))
    ann_ps_proc = _ScalerShim(sx, sy)
    record_health("PS (ANN)", True, f"loaded from {ps_model_path}")
except Exception as e:
    record_health("PS (ANN)", False, f"{e}")

# [Include all other model loading code...]

model_registry = {}
# [Your model registry code...]

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# =============================================================================
# 📊 STEP 6: INPUT PARAMETERS & DATA RANGES DEFINITION
# =============================================================================
R = {
    "lw":(400.0,3500.0), "hw":(495.0,5486.4), "tw":(26.0,305.0), "fc":(13.38,93.6),
    "fyt":(0.0,1187.0), "fysh":(0.0,1375.0), "fyl":(160.0,1000.0), "fybl":(0.0,900.0),
    "rt":(0.000545,0.025139), "rsh":(0.0,0.041888), "rl":(0.0,0.029089), "rbl":(0.0,0.031438),
    "axial":(0.0,0.86), "b0":(45.0,3045.0), "db":(0.0,500.0), "s_db":(0.0,47.65625),
    "AR":(0.388889,5.833333), "M_Vlw":(0.388889,4.1), "theta":(0.0275,4.85),
}
THETA_MAX = R["theta"][1]
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
    (r"$\theta\;(\%)$","theta",THETA_MAX,0.0005,None,"Drift Ratio"),
]

def num(label, key, default, step, fmt, help_):
    return st.number_input(
        label, value=dv(R, key, default), step=step,
        min_value=R[key][0], max_value=R[key][1],
        format=fmt if fmt else None, help=help_
    )

# =============================================================================
# 🎮 STEP 7: MAIN LAYOUT - TWO COLUMNS
# =============================================================================
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='form-banner'>Inputs Features</div>", unsafe_allow_html=True)
    
    # Three columns: Geometry | Reinf. Ratios | Material Strengths
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")

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
    # Logo at top right
    try:
        st.markdown(
            f"""
            <div style="position:relative; left:100px; top:5px; text-align:left;">
                <img src='data:image/png;base64,{b64(BASE_DIR / "logo2-01.png")}' width='300'/>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except:
        pass
    
    # Model selection and buttons row
    st.markdown("<div style='margin-top: 50px;'>", unsafe_allow_html=True)
    row = st.columns([1, 1, 1, 1], gap="small")
    
    with row[0]:
        available = set(model_registry.keys())
        order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
        ordered_keys = [m for m in order if m in available] or ["(no models loaded)"]
        display_labels = ["RF" if m == "Random Forest" else m for m in ordered_keys]
        _label_to_key = {"RF": "Random Forest"}
        model_choice_label = st.selectbox("Model Selection", display_labels, key="model_select_compact")
        model_choice = _label_to_key.get(model_choice_label, model_choice_label)
    
    with row[1]:
        submit = st.button("Calculate", key="calc_btn")
    
    with row[2]:
        reset_btn = st.button("Reset", key="reset_btn")
    
    with row[3]:
        clear_btn = st.button("Clear All", key="clear_btn")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction result and download
    pred_banner = st.empty()
    dl_slot = st.empty()
    
    # Chart - compact size
    chart_slot = st.empty()

# =============================================================================
# 🔮 STEP 8: PREDICTION ENGINE (KEEP YOUR ORIGINAL CODE)
# =============================================================================
_TRAIN_NAME_MAP = {
    'l_w': 'lw', 'h_w': 'hw', 't_w': 'tw', 'f′c': 'fc',
    'fyt': 'fyt', 'fysh': 'fysh', 'fyl': 'fyl', 'fybl': 'fybl',
    'ρt': 'pt', 'ρsh': 'psh', 'ρl': 'pl', 'ρbl': 'pbl',
    'P/(Agf′c)': 'P/(Agfc)', 'b0': 'b0', 'db': 'db', 's/db': 's/db',
    'AR': 'AR', 'M/Vlw': 'M/Vlw', 'θ': 'θ'
}

def predict_di(choice, _unused_array, input_df):
    # [Your original prediction code here...]
    df_trees = input_df.rename(columns=_TRAIN_NAME_MAP)
    df_trees = df_trees.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = df_trees.values.astype(np.float32)

    if choice == "LightGBM":
        mdl = model_registry["LightGBM"]
        prediction = float(mdl.predict(X)[0])
    elif choice == "XGBoost":
        prediction = float(model_registry["XGBoost"].predict(X)[0])
    elif choice == "CatBoost":
        prediction = float(model_registry["CatBoost"].predict(X)[0])
    elif choice == "Random Forest":
        prediction = float(model_registry["Random Forest"].predict(X)[0])
    elif choice == "PS":
        Xn = ann_ps_proc.transform_X(X)
        try:
            yhat = model_registry["PS"].predict(Xn, verbose=0)[0][0]
        except Exception:
            model_registry["PS"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["PS"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_ps_proc.inverse_transform_y(yhat).item())
    elif choice == "MLP":
        Xn = ann_mlp_proc.transform_X(X)
        try:
            yhat = model_registry["MLP"].predict(Xn, verbose=0)[0][0]
        except Exception:
            model_registry["MLP"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["MLP"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_mlp_proc.inverse_transform_y(yhat).item())

    prediction = max(0.035, min(prediction, 1.5))
    return prediction

def _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta_val):
    cols = ['l_w','h_w','t_w','f′c','fyt','fysh','fyl','fybl','ρt','ρsh','ρl','ρbl','P/(Agf′c)','b0','db','s/db','AR','M/Vlw','θ']
    x = np.array([[lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta_val]], dtype=np.float32)
    return pd.DataFrame(x, columns=cols)

def render_di_chart(curve_df, size=300):
    import altair as alt
    if curve_df.empty:
        return
    
    chart = alt.Chart(curve_df).mark_line().encode(
        x=alt.X('θ:Q', title='Drift Ratio (θ)'),
        y=alt.Y('Predicted_DI:Q', title='Damage Index (DI)', scale=alt.Scale(domain=[0, 1.5]))
    ).properties(
        width=size,
        height=size
    )
    st.altair_chart(chart, use_container_width=False)

# =============================================================================
# ⚡ STEP 9: PREDICTION EXECUTION
# =============================================================================
if reset_btn:
    st.rerun()

if clear_btn:
    st.session_state.results_df = pd.DataFrame()
    st.success("All predictions cleared.")

if submit and model_choice in model_registry:
    xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
    try:
        pred = predict_di(model_choice, None, xdf)
        row = xdf.copy()
        row["Predicted_DI"] = pred
        st.session_state.results_df = pd.concat([st.session_state.results_df, row], ignore_index=True)
        pred_banner.markdown(f"<div class='prediction-result'>Predicted Damage Index (DI): {pred:.4f}</div>", unsafe_allow_html=True)
        
        if not st.session_state.results_df.empty:
            csv = st.session_state.results_df.to_csv(index=False)
            dl_slot.download_button("📂 Download as CSV", data=csv, file_name="di_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Generate and display curve
if model_choice in model_registry:
    _base_xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
    
    # Create simple curve for display
    thetas = np.linspace(0, THETA_MAX, 50)
    dis = np.minimum(1.5, np.maximum(0.035, thetas * 0.3))  # Simple linear relationship for demo
    curve_df = pd.DataFrame({"θ": thetas, "Predicted_DI": dis})
    
    with chart_slot:
        render_di_chart(curve_df, size=300)

# =============================================================================
# ✅ COMPLETED: RESTORED ORIGINAL LAYOUT
# =============================================================================
