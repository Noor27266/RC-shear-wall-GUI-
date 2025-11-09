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

# ====== COMPACT SIZING ======
SCALE_UI = 0.25  # Much smaller scaling

s = lambda v: int(round(v * SCALE_UI))

FS_TITLE   = s(60)   # Smaller title
FS_SECTION = s(40)   # Smaller section headers
FS_LABEL   = s(35)   # Smaller labels
FS_UNITS   = s(20)   # Smaller units
FS_INPUT   = s(20)   # Smaller input text
FS_SELECT  = s(25)   # Smaller select
FS_BUTTON  = s(16)   # Smaller buttons
FS_BADGE   = s(20)   # Smaller badge
FS_RECENT  = s(14)   # Smaller recent
INPUT_H    = max(28, int(FS_INPUT * 1.8))  # Smaller input height

# header logo default height
DEFAULT_LOGO_H = 40  # Smaller logo

PRIMARY   = "#8E44AD"
SECONDARY = "#f9f9f9"
INPUT_BG     = "#ffffff"
INPUT_BORDER = "#e6e9f2"
LEFT_BG      = "#e0e4ec"

# =============================================================================
# 🎨 STEP 3.1: COMPREHENSIVE CSS STYLING & THEME SETUP
# =============================================================================

# Add viewport meta tag for zoom stability first
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

css(f"""
<style>
  /* COMPLETELY LOCK THE INTERFACE IN PLACE */
  html, body, .stApp {{
    overflow: hidden !important;
    position: fixed !important;
    width: 100vw !important;
    height: 100vh !important;
    transform: none !important;
    left: 0 !important;
    top: 0 !important;
    zoom: 1 !important;
  }}

  /* FORCE ENTIRE INTERFACE TO THE RIGHT - COMPACT */
  .main .block-container {{
    position: absolute !important;
    left: 200px !important;
    top: 0 !important;
    width: calc(100vw - 200px) !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 10px !important;
    transform: none !important;
    overflow: auto !important;
  }}

  /* Remove all Streamlit default centering and spacing */
  .stApp {{
    align-items: flex-start !important;
    justify-content: flex-start !important;
    padding: 0 !important;
    margin: 0 !important;
  }}

  /* Remove ALL top spacing */
  header[data-testid="stHeader"] {{ 
    display: none !important; 
    height: 0 !important;
  }}

  section.main > div.block-container {{
    padding-top: 5px !important;
    margin-top: 0 !important;
  }}

  /* Lock all containers in place */
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewContainer"] > div,
  section.main {{
    position: static !important;
    transform: none !important;
    left: 0 !important;
    top: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
  }}

  /* COMPACT TITLE STYLING */
  .page-header__title {{
    font-size:{FS_TITLE}px !important;
    font-weight:700 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
  }}

  .section-header {{
    font-size:{FS_SECTION}px !important;
    font-weight:600; 
    margin: 0.2rem 0 !important;
    padding: 0 !important;
  }}

  .stNumberInput label, .stSelectbox label {{
    font-size:{FS_LABEL}px !important; 
    font-weight:600;
    margin-bottom: 2px !important;
  }}
  .stNumberInput label .katex,
  .stSelectbox label .katex {{ 
    font-size:{FS_LABEL}px !important; 
    line-height:1.1 !important; 
  }}

  .stNumberInput label .katex .mathrm,
  .stSelectbox  label .katex .mathrm {{ 
    font-size:{FS_UNITS}px !important; 
  }}

  div[data-testid="stNumberInput"] input[type="number"],
  div[data-testid="stNumberInput"] input[type="text"] {{
      font-size:{FS_INPUT}px !important;
      height:{INPUT_H}px !important;
      line-height:{INPUT_H - 8}px !important;
      font-weight:500 !important;
      padding: 6px 8px !important;
  }}

  div[data-testid="stNumberInput"] [data-baseweb*="input"] {{
      background:{INPUT_BG} !important;
      border:1px solid {INPUT_BORDER} !important;
      border-radius:8px !important;
      box-shadow:0 1px 2px rgba(16,24,40,.06) !important;
  }}

  /* Select font sizes are tied to FS_SELECT */
  .stSelectbox [role="combobox"],
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child,
  div[data-testid="stSelectbox"] div[role="listbox"],
  div[data-testid="stSelectbox"] div[role="option"] {{
      font-size:{FS_SELECT}px !important;
  }}

  /* Buttons use FS_BUTTON, no wrapping */
  div.stButton > button {{
    font-size:{FS_BUTTON}px !important;
    height:{max(32, int(round(FS_BUTTON*1.3)))}px !important;
    line-height:{max(28, int(round(FS_BUTTON*1.1)))}px !important;
    white-space:nowrap !important;
    color:#fff !important;
    font-weight:600; 
    border:none !important; 
    border-radius:6px !important;
    background:#4CAF50 !important;
    padding: 0 12px !important;
  }}

  button[key="calc_btn"] {{ background:#4CAF50 !important; }}
  button[key="reset_btn"] {{ background:#2196F3 !important; }}
  button[key="clear_btn"] {{ background:#f44336 !important; }}

  .form-banner {{
    text-align:center;
    background: linear-gradient(90deg, #0E9F6E, #84CC16);
    color: #fff;
    padding: 0.3rem 0.5rem !important;
    border-radius:6px;
    font-weight:600;
    font-size:{FS_SECTION}px;
    margin: 0.2rem 0 !important;
  }}

  .prediction-result {{
    font-size:{FS_BADGE}px !important; 
    font-weight:600; 
    color:#2e86ab;
    background:#f1f3f4; 
    padding:0.4rem; 
    border-radius:4px; 
    text-align:center; 
    margin-top:0.4rem;
  }}

  #compact-form{{ max-width:800px; margin:0 auto; }}
  #compact-form [data-testid="stHorizontalBlock"]{{ gap:0.3rem; flex-wrap:nowrap; }}
  #compact-form [data-testid="column"]{{ width:180px; max-width:180px; flex:0 0 180px; padding:0; }}
  #compact-form [data-testid="stNumberInput"],
  #compact-form [data-testid="stNumberInput"] *{{ max-width:none; box-sizing:border-box; }}
  #compact-form [data-testid="stNumberInput"]{{ display:inline-flex; width:auto; min-width:0; flex:0 0 auto; margin-bottom:0.2rem; }}

  .block-container [data-testid="stHorizontalBlock"] > div:has(.form-banner) {{
      background:{LEFT_BG} !important;
      border-radius:8px !important;
      box-shadow:0 1px 2px rgba(0,0,0,.1) !important;
      padding:12px !important;
      margin: 5px 0 !important;
  }}

  /* COMPACT HEADER POSITIONING */
  .page-header {{ 
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 !important;
    padding: 0 1rem !important;
    width: 100%;
    position: relative;
    left: 0 !important;
    transform: none !important;
    height: 50px !important;
  }}
  
  .page-header__title {{
    font-size:{FS_TITLE}px !important;
    font-weight:700 !important; 
    margin:0 !important;
    flex: 1;
    margin-left: 10px !important;
    margin-top: 5px !important;
    transform: none !important;
  }}

  .page-header__logo {{
    height:35px !important; 
    width:auto !important; 
    position: absolute;
    right: 20px !important;
    top: 8px !important;
    z-index: 1000;
    transform: none !important;
  }}

  .page-header-outer {{
    width: 100%;
    margin-left: 0 !important;
    transform: none !important;
    position: relative;
    height: 50px !important;
    margin-bottom: 5px !important;
  }}

  /* Compact column spacing */
  [data-testid="column"] {{
    padding: 0 5px !important;
  }}

  /* Prevent any element from moving on zoom */
  [data-testid="column"], 
  [data-testid="stHorizontalBlock"],
  .stNumberInput,
  .stSelectbox {{
    position: relative !important;
    left: 0 !important;
    transform: none !important;
  }}
</style>
""")

# Additional CSS to remove all default Streamlit spacing
st.markdown("""
<style>
/* Remove ALL Streamlit default spacing and centering */
div.stApp { 
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Hide Streamlit's small +/- buttons on number inputs */
div[data-testid="stNumberInput"] button { display: none !important; }

/* Also hide browser numeric spinners for consistency */
div[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
div[data-testid="stNumberInput"] input::-webkit-inner-spin-button { 
    -webkit-appearance: none; 
    margin: 0; 
}
div[data-testid="stNumberInput"] input[type=number] { 
    -moz-appearance: textfield; 
}

/* Increase the width of the Predicted Damage Index (DI) box */
.prediction-result {
  width: auto !important;
  max-width: 200px !important;
  padding: 3px 8px !important;
  font-size: 0.8em !important;
  white-space: nowrap !important;
  margin-right: 10px !important;
}

/* Move the Download CSV button closer to the DI box */
div[data-testid="stDownloadButton"] {
  display: inline-block !important;
  margin-left:-80px !important;
}
div[data-testid="stDownloadButton"] button {
  white-space: nowrap !important;
  padding: 2px 6px !important;
  font-size: 7px !important;
  height: auto !important;
  line-height: 1 !important;
}

/* Model selection box styling */
div[data-testid="stSelectbox"] [data-baseweb="select"] {
    width: 100% !important;
    height: 25px !important;
}

div[data-testid="stSelectbox"] > div > div {
    height: 90px !important;
    line-height: 25px !important;
}

/* Reduce spacing in columns */
[data-testid="stHorizontalBlock"] {
    gap: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 🏷️ STEP 4: DYNAMIC HEADER & LOGO POSITIONING
# =============================================================================
try:
    _logo_path = BASE_DIR / "TJU logo.png"
    _b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii") if _logo_path.exists() else ""
except Exception:
    _b64 = ""

st.markdown(f"""
<style>
  .page-header {{ 
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 !important;
    padding: 0 1rem !important;
    width: 100%;
    position: relative;
    left: 0 !important;
    transform: none !important;
    height: 45px !important;
  }}
  
  .page-header__title {{
    font-size:{FS_TITLE}px !important;
    font-weight:700 !important; 
    margin:0 !important;
    flex: 1;
    margin-left: 10px !important;
    margin-top: 5px !important;
    transform: none !important;
  }}

  .page-header__logo {{
    height:35px !important; 
    width:auto !important; 
    position: absolute;
    right: 20px !important;
    top: 5px !important;
    z-index: 1000;
    transform: none !important;
  }}

  .page-header-outer {{
    width: 100%;
    margin-left: 0 !important;
    transform: none !important;
    position: relative;
    height: 45px !important;
    margin-bottom: 5px !important;
  }}
</style>

<div class="page-header-outer">
  <div class="page-header">
    <div class="page-header__title">Predict Damage index (DI) for RC Shear Walls</div>
    {f'<img class="page-header__logo" alt="Logo" src="data:image/png;base64,{_b64}" />' if _b64 else ''}
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# 🤖 STEP 5: MACHINE LEARNING MODEL LOADING & HEALTH CHECKING
# =============================================================================
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

ann_mlp_model = None; ann_mlp_proc = None
try:
    mlp_model_path = pfind(["ANN_MLP_Model.keras", "ANN_MLP_Model.h5"])
    ann_mlp_model = _load_keras_model(mlp_model_path)
    sx = joblib.load(pfind(["ANN_MLP_Scaler_X.save","ANN_MLP_Scaler_X.pkl","ANN_MLP_Scaler_X.joblib"]))
    sy = joblib.load(pfind(["ANN_MLP_Scaler_y.save","ANN_MLP_Scaler_y.pkl","ANN_MLP_Scaler_y.joblib"]))
    ann_mlp_proc = _ScalerShim(sx, sy)
    record_health("MLP (ANN)", True, f"loaded from {mlp_model_path}")
except Exception as e:
    record_health("MLP (ANN)", False, f"{e}")

rf_model = None
try:
    rf_path = pfind([
        "random_forest_model.pkl", "random_forest_model.joblib",
        "rf_model.pkl", "RF_model.pkl",
        "Best_RF_Model.json", "best_rf_model.json", "RF_model.json"
    ])
    try:
        rf_model = joblib.load(rf_path)
        record_health("Random Forest", True, f"loaded with joblib from {rf_path}")
    except Exception as e_joblib:
        try:
            import skops.io as sio
            rf_model = sio.load(rf_path, trusted=True)
            record_health("Random Forest", True, f"loaded via skops from {rf_path}")
        except Exception as e_skops:
            record_health("Random Forest", False, f"RF load failed for {rf_path} (joblib: {e_joblib}) (skops: {e_skops})")
except Exception as e:
    record_health("Random Forest", False, str(e))

xgb_model = None
try:
    xgb_path = pfind(["XGBoost_trained_model_for_DI.json","Best_XGBoost_Model.json","xgboost_model.json"])
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model(xgb_path)
    record_health("XGBoost", True, f"loaded from {xgb_path}")
except Exception as e:
    record_health("XGBoost", False, str(e))

cat_model = None
try:
    cat_path = pfind(["CatBoost.cbm","Best_CatBoost_Model.cbm","catboost.cbm"])
    cat_model = catboost.CatBoostRegressor(); cat_model.load_model(cat_path)
    record_health("CatBoost", True, f"loaded from {cat_path}")
except Exception as e:
    record_health("CatBoost", False, f"{e}")

def load_lightgbm_flex():
    try:
        p = pfind(["LightGBM_model.txt","Best_LightGBM_Model.txt","LightGBM_model.bin","LightGBM_model.pkl","LightGBM_model.joblib","LightGBM_model"])
    except Exception:
        raise FileNotFoundError("No LightGBM model file found.")
    try: return lgb.Booster(model_file=str(p)), "booster", p
    except Exception:
        try: return joblib.load(p), "sklearn", p
        except Exception as e:
            raise e

try:
    lgb_model, lgb_kind, lgb_path = load_lightgbm_flex()
    record_health("LightGBM", True, f"loaded as {lgb_kind} from {lgb_path}")
except Exception as e:
    lgb_model = None; record_health("LightGBM", False, str(e))

model_registry = {}
for name, ok, *_ in health:
    if not ok: continue
    if name == "XGBoost" and xgb_model is not None: model_registry["XGBoost"] = xgb_model
    elif name == "LightGBM" and lgb_model is not None: model_registry["LightGBM"] = lgb_model
    elif name == "CatBoost" and cat_model is not None: model_registry["CatBoost"] = cat_model
    elif name == "PS (ANN)" and ann_ps_model is not None: model_registry["PS"] = ann_ps_model
    elif name == "MLP (ANN)" and ann_mlp_model is not None: model_registry["MLP"] = ann_mlp_model
    elif name == "Random Forest" and rf_model is not None: model_registry["Random Forest"] = rf_model

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

left, right = st.columns([1, 1], gap="medium")

with left:
    st.markdown("<div class='form-banner'>Inputs Features</div>", unsafe_allow_html=True)
    css("<div id='compact-form'>")

    # ⬇️ Three columns: Geometry | Reinf. Ratios | Material Strengths
    c1, c2, c3 = st.columns([1, 1, 1], gap="small")

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

    css("</div>")

# =============================================================================
# 🎮 STEP 7: RIGHT PANEL - CONTROLS & INTERACTION ELEMENTS
# =============================================================================
HERO_X, HERO_Y, HERO_W = 50, 0, 200
CHART_W = 280

with right:
    st.markdown(
        f"""
        <div style="position:relative; left:{int(HERO_X)}px; top:{int(HERO_Y)}px; text-align:left; margin-bottom: 10px;">
            <img src='data:image/png;base64,{b64(BASE_DIR / "logo2-01.png")}' width='{int(HERO_W)}'/>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(""" 
    <style>
    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        border: 1px solid #e6e9f2 !important; box-shadow: none !important; background: #fff !important;
    }
    div[data-testid="stSelectbox"] > div > div { height: 40px !important; display:flex !important; align-items:center !important; margin-top: -0px; }
    div[data-testid="stSelectbox"] label p { font-size: {FS_LABEL}px !important; color: black !important; font-weight: bold !important; }
    [data-baseweb="select"] *, [data-baseweb="popover"] *, [data-baseweb="menu"] * { color: black !important; background-color: #D3D3D3 !important; font-size: {FS_SELECT}px !important; }
    div[role="option"] { color: black !important; font-size: {FS_SELECT}px !important; }
    div.stButton > button { height: {max(32, int(round(FS_BUTTON*1.3)))}px !important; display:flex; align-items:center; justify-content:center; }
    #action-row { display:flex; align-items:center; gap: 1px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div id='action-row'>", unsafe_allow_html=True)
    row = st.columns([0.8, 2.1, 2.1, 2.1], gap="small")

    with row[0]:
        available = set(model_registry.keys())
        order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
        ordered_keys = [m for m in order if m in available] or ["(no models loaded)"]
        display_labels = ["RF" if m == "Random Forest" else m for m in ordered_keys]
        _label_to_key = {"RF": "Random Forest"}
        model_choice_label = st.selectbox("Model Selection", display_labels, key="model_select_compact")
        model_choice = _label_to_key.get(model_choice_label, model_choice_label)

    with row[1]:
        st.markdown("<div id='three-btns' style='margin-top:25px;'>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1, 1, 1.2], gap="small")
        with b1:
            submit = st.button("Calculate", key="calc_btn")
        with b2:
            if st.button("Reset", key="reset_btn"):
                st.rerun()
        with b3:
            if st.button("Clear All", key="clear_btn"):
                st.session_state.results_df = pd.DataFrame()
                st.success("All predictions cleared.")
        st.markdown("</div>", unsafe_allow_html=True)

    badge_col, dl_col, _spacer = st.columns([5, 3.0, 7], gap="small")
    with badge_col:
        pred_banner = st.empty()
    with dl_col:
        dl_slot = st.empty()
    if not st.session_state.results_df.empty:
        csv = st.session_state.results_df.to_csv(index=False)
        dl_slot.download_button("📂 Download as CSV", data=csv, file_name="di_predictions.csv", mime="text/csv", use_container_width=False, key="dl_csv_main")

    col1, col2 = st.columns([0.01, 20])
    with col2:
        chart_slot = st.empty()

# =============================================================================
# 🔮 STEP 8: PREDICTION ENGINE & CURVE GENERATION UTILITIES
# =============================================================================
_TRAIN_NAME_MAP = {
    'l_w': 'lw', 'h_w': 'hw', 't_w': 'tw', 'f′c': 'fc',
    'fyt': 'fyt', 'fysh': 'fysh', 'fyl': 'fyl', 'fybl': 'fybl',
    'ρt': 'pt', 'ρsh': 'psh', 'ρl': 'pl', 'ρbl': 'pbl',
    'P/(Agf′c)': 'P/(Agfc)', 'b0': 'b0', 'db': 'db', 's/db': 's/db',
    'AR': 'AR', 'M/Vlw': 'M/Vlw', 'θ': 'θ'
}
_TRAIN_COL_ORDER = ['lw','hw','tw','fc','fyt','fysh','fyl','fybl','pt','psh','pl','pbl','P/(Agfc)','b0','db','s/db','AR','M/Vlw','θ']

def _df_in_train_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_TRAIN_NAME_MAP).reindex(columns=_TRAIN_COL_ORDER)

def predict_di(choice, _unused_array, input_df):
    df_trees = _df_in_train_order(input_df)
    df_trees = df_trees.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = df_trees.values.astype(np.float32)

    if choice == "LightGBM":
        mdl = model_registry["LightGBM"]
        prediction = float(mdl.predict(X)[0])
    if choice == "XGBoost":
        prediction = float(model_registry["XGBoost"].predict(X)[0])
    if choice == "CatBoost":
        prediction = float(model_registry["CatBoost"].predict(X)[0])
    if choice == "Random Forest":
        prediction = float(model_registry["Random Forest"].predict(X)[0])
    if choice == "PS":
        Xn = ann_ps_proc.transform_X(X)
        try:
            yhat = model_registry["PS"].predict(Xn, verbose=0)[0][0]
        except Exception:
            model_registry["PS"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["PS"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_ps_proc.inverse_transform_y(yhat).item())
    if choice == "MLP":
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

def _sweep_curve_df(model_choice, base_df, theta_max=THETA_MAX, step=0.1):
    if model_choice not in model_registry:
        return pd.DataFrame(columns=["θ","Predicted_DI"])
    thetas = np.round(np.arange(0.0, theta_max + 1e-9, step), 2)
    rows = []
    for th in thetas:
        df = base_df.copy()
        df.loc[:, 'θ'] = float(th)
        di = predict_di(model_choice, None, df)
        di = max(0.035, min(di, 1.5))
        rows.append({"θ": float(th), "Predicted_DI": float(di)})
    return pd.DataFrame(rows)

def render_di_chart(results_df: pd.DataFrame, curve_df: pd.DataFrame,
                    theta_max: float = THETA_MAX, di_max: float = 1.5, size: int = 280):
    import altair as alt
    selection = alt.selection_point(name='select', fields=['θ', 'Predicted_DI'], nearest=True, on='mouseover', empty=False, clear='mouseout')
    AXIS_LABEL_FS = 12; AXIS_TITLE_FS = 14; TICK_SIZE = 4; TITLE_PAD = 8; LABEL_PAD = 4
    base_axes_df = pd.DataFrame({"θ": [0.0, theta_max], "Predicted_DI": [0.0, 0.0]})
    x_ticks = np.linspace(0.0, theta_max, 5).round(2)

    axes_layer = (
        alt.Chart(base_axes_df).mark_line(opacity=0).encode(
            x=alt.X("θ:Q", title="Drift Ratio (θ)", scale=alt.Scale(domain=[0, theta_max], nice=False, clamp=True),
                    axis=alt.Axis(values=list(x_ticks), labelFontSize=AXIS_LABEL_FS, titleFontSize=AXIS_TITLE_FS,
                                  labelPadding=LABEL_PAD, titlePadding=TITLE_PAD, tickSize=TICK_SIZE, labelLimit=1000,
                                  labelFlush=True, labelFlushOffset=0)),
            y=alt.Y("Predicted_DI:Q", title="Damage Index (DI)", scale=alt.Scale(domain=[0, di_max], nice=False, clamp=True),
                    axis=alt.Axis(values=[0.0, 0.2, 0.5, 1.0, 1.5], labelFontSize=AXIS_LABEL_FS, titleFontSize=AXIS_TITLE_FS,
                                  labelPadding=LABEL_PAD, titlePadding=TITLE_PAD, tickSize=TICK_SIZE, labelLimit=1000,
                                  labelFlush=True, labelFlushOffset=0)),
        ).properties(width=size, height=size)
    )

    curve = curve_df if (curve_df is not None and not curve_df.empty) else pd.DataFrame({"θ": [], "Predicted_DI": []})
    line_layer = alt.Chart(curve).mark_line(strokeWidth=2).encode(x="θ:Q", y="Predicted_DI:Q").properties(width=size, height=size)

    k = 3
    if not curve.empty:
        curve_points = curve.iloc[::k].copy()
    else:
        curve_points = pd.DataFrame({"θ": [], "Predicted_DI": []})

    points_layer = alt.Chart(curve_points).mark_circle(size=40, opacity=0.7).encode(
        x="θ:Q", y="Predicted_DI:Q",
        tooltip=[alt.Tooltip("θ:Q", title="Drift Ratio (θ)", format=".2f"),
                 alt.Tooltip("Predicted_DI:Q", title="Predicted DI", format=".4f")]
    ).add_params(selection)

    rules_layer = alt.Chart(curve).mark_rule(color='red', strokeWidth=2).encode(x="θ:Q", y="Predicted_DI:Q").transform_filter(selection)
    text_layer = alt.Chart(curve).mark_text(align='left', dx=8, dy=-8, fontSize=12, fontWeight='bold', color='red').encode(
        x="θ:Q", y="Predicted_DI:Q", text=alt.Text("Predicted_DI:Q", format=".4f")
    ).transform_filter(selection)

    chart = (alt.layer(axes_layer, line_layer, points_layer, rules_layer, text_layer)
             .configure_view(strokeWidth=0)
             .configure_axis(domain=True, ticks=True)
             .configure(padding={"left": 4, "right": 4, "top": 4, "bottom": 4}))
    chart_html = chart.to_html()
    chart_html = chart_html.replace('</style>',
        '</style><style>.vega-embed .vega-tooltip, .vega-embed .vega-tooltip * { font-size: 12px !important; font-weight: bold !important; background: #000 !important; color: #fff !important; padding: 8px !important; }</style>')
    st.components.v1.html(chart_html, height=size + 80)

# =============================================================================
# ⚡ STEP 9: PREDICTION EXECUTION & REAL-TIME VISUALIZATION
# =============================================================================
_order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
_label_to_key = {"RF": "Random Forest"}

def _pick_default_model():
    for m in _order:
        if m in model_registry:
            return m
    return None

if 'model_choice' not in locals():
    _label = (st.session_state.get("model_select_compact")
              or st.session_state.get("model_select"))
    if _label is not None:
        model_choice = _label_to_key.get(_label, _label)
    else:
        model_choice = _pick_default_model()

if (model_choice is None) or (model_choice not in model_registry):
    st.error("No trained model is available. Please check the Model Selection on the right.")
else:
    if 'submit' in locals() and submit:
        xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
        try:
            pred = predict_di(model_choice, None, xdf)
            row = xdf.copy(); row["Predicted_DI"] = pred
            st.session_state.results_df = pd.concat([st.session_state.results_df, row], ignore_index=True)
            pred_banner.markdown(f"<div class='prediction-result'>Predicted Damage Index (DI): {pred:.4f}</div>", unsafe_allow_html=True)
            csv = st.session_state.results_df.to_csv(index=False)
            dl_slot.download_button("📂 Download as CSV", data=csv, file_name="di_predictions.csv",
                                    mime="text/csv", use_container_width=False, key="dl_csv_after_submit")
        except Exception as e:
            st.error(f"Prediction failed for {model_choice}: {e}")

    _base_xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
    _curve_df = _sweep_curve_df(model_choice, _base_xdf, theta_max=THETA_MAX, step=0.1)

try:
    _slot = chart_slot
except NameError:
    _slot = st.empty()

with right:
    with _slot:
        render_di_chart(st.session_state.results_df, _curve_df, theta_max=THETA_MAX, di_max=1.5, size=CHART_W)

# =============================================================================
# ✅ COMPLETED: RC SHEAR WALL DI ESTIMATOR APPLICATION
# =============================================================================
