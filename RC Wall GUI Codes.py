# -*- coding: utf-8 -*-

DOC_NOTES = """
RC Shear Wall Damage Index (DI) Estimator — compact, same logic/UI
"""

# =============================================================================
# Step #1: Core imports and TensorFlow backend guard
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

from tensorflow.keras.models import load_model

# --- session defaults (prevents AttributeError on first run) ---
st.session_state.setdefault("results_df", pd.DataFrame())

# =============================================================================
# Small helpers
# =============================================================================
css = lambda s: st.markdown(s, unsafe_allow_html=True)
def b64(path: Path) -> str: return base64.b64encode(path.read_bytes()).decode("ascii")
def dv(R, key, proposed): lo, hi = R[key]; return float(max(lo, min(proposed, hi)))

# ---------- path helper (so PS/MLP/RF actually load) ----------
BASE_DIR = Path(__file__).resolve().parent
def pfind(candidates):
    """
    Return the first existing file path among `candidates`.
    Search order:
      1) exact paths
      2) alongside this script (BASE_DIR)
      3) current working directory
      4) /mnt/data (where uploaded files live)
      5) one-level subdirs under BASE_DIR and /mnt/data
      6) recursive glob fallback
    """
    # exact paths first
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

    # one-level subdirs under BASE_DIR and /mnt/data
    for root in [BASE_DIR, Path("/mnt/data")]:
        if not root.exists():
            continue
        for sub in root.iterdir():
            if sub.is_dir():
                for c in candidates:
                    p = sub / c
                    if p.exists():
                        return p

    # glob fallbacks
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
# Step #2: Page config + COLORS + font knobs
# =============================================================================
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

FS_TITLE   = 50
FS_SECTION = 35
FS_LABEL   = 30
FS_UNITS   = 18
FS_INPUT   = 20
FS_SELECT  = 50
FS_BUTTON  = 55
FS_BADGE   = 25
FS_RECENT  = 16
INPUT_H    = max(32, int(FS_INPUT * 2.1))

PRIMARY   = "#8E44AD"
SECONDARY = "#f9f9f9"

INPUT_BG     = "#ffffff"
INPUT_BORDER = "#e6e9f2"
LEFT_BG      = "#e0e4ec"

# =============================================================================
# Step #2.1: Global UI CSS (layout, fonts, inputs, theme)
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
  .stNumberInput label .katex,
  .stSelectbox label .katex {{ font-size:{FS_LABEL}px !important; line-height:1.2 !important; }}
  .stNumberInput label .katex .fontsize-ensurer,
  .stSelectbox label .katex .fontsize-ensurer {{ font-size:1em !important; }}

  .stNumberInput label .katex .mathrm,
  .stSelectbox  label .katex .mathrm {{ font-size:{FS_UNITS}px !important; }}

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
      box-shadow:0 1px 2px rgba(16,24,40,.06) !important;
      transition:border-color .15s ease, box-shadow .15s ease !important;
  }}
  div[data-testid="stNumberInput"] [data-baseweb*="input"]:hover {{
      border-color:#d6dced !important;
  }}
  div[data-testid="stNumberInput"] [data-baseweb*="input"]:focus-within {{
      border-color:{PRIMARY} !important;
      box-shadow:0 0 0 3px rgba(106,17,203,.15) !important;
  }}

  div[data-testid="stNumberInput"] button {{
      background:#ffffff !important;
      border:1px solid {INPUT_BORDER} !important;
      border-radius:10px !important;
      box-shadow:0 1px 1px rgba(16,24,40,.05) !important;
  }}
  div[data-testid="stNumberInput"] button:hover {{
      border-color:#cbd3e5 !important;
  }}

  .stSelectbox [role="combobox"] {{ font-size:{FS_SELECT}px !important; }}

  div.stButton > button {{
    font-size:{FS_BUTTON}px !important;
    height:40px !important;
    color:#fff !important;
    font-weight:700; border:none !important; border-radius:8px !important;
    background: #4CAF50 !important;
  }}
  div.stButton > button:hover {{ filter: brightness(0.95); }}

  button[key="calc_btn"] {{ background: #4CAF50 !important; }}
  button[key="reset_btn"] {{ background: #2196F3 !important; }}
  button[key="clear_btn"] {{ background: #f44336 !important; }}

  .form-banner {{
    text-align:center;
    background: linear-gradient(90deg, #0E9F6E, #84CC16);
    color: #fff;
    padding:.45rem .75rem;
    border-radius:10px;
    font-weight:800;
    font-size:{FS_SECTION + 4}px;
    margin:.1rem 0 !important;
    transform: translateY(-10px);
  }}

  .prediction-result {{
    font-size:{FS_BADGE}px !important; font-weight:700; color:#2e86ab;
    background:#f1f3f4; padding:.6rem; border-radius:6px; text-align:center; margin-top:.6rem;
  }}
  .recent-box {{
    font-size:{FS_RECENT}px !important; background:#f8f9fa; padding:.5rem; margin:.25rem 0;
    border-radius:5px; border-left:4px solid #4CAF50; font-weight:600; display:inline-block;
  }}

  #compact-form{{ max-width:900px; margin:0 auto; }}
  #compact-form [data-testid="stHorizontalBlock"]{{ gap:.5rem; flex-wrap:nowrap; }}
  #compact-form [data-testid="column"]{{ width:200px; max-width:200px; flex:0 0 200px; padding:0; }}
  #compact-form [data-testid="stNumberInput"],
  #compact-form [data-testid="stNumberInput"] *{{ max-width:none; box-sizing:border-box; }}
  #compact-form [data-testid="stNumberInput"]{{ display:inline-flex; width:auto; min-width:0; flex:0 0 auto; margin-bottom:.35rem; }}
  #button-row {{ display:flex; gap:30px; margin:10px 0 6px 0; align-items:center; }}

  .block-container [data-testid="stHorizontalBlock"] > div:has(.form-banner) {{
      background:{LEFT_BG} !important;
      border-radius:12px !important;
      box-shadow:0 1px 3px rgba(0,0,0,.1) !important;
      padding:16px !important;
  }}

  .left-panel {{
      background:{LEFT_BG} !important;
      border-radius:12px !important;
      box-shadow:0 1px 3px rgba(0,0,0,.1) !important;
      padding:16px !important;
  }}

  [data-baseweb="popover"], [data-baseweb="tooltip"],
  [data-baseweb="popover"] > div, [data-baseweb="tooltip"] > div {{
      background: #000 !important; color: #fff !important; border-radius: 8px !important;
      padding: 6px 10px !important; font-size: 24px !important; font-weight: 500 !important;
  }}
  [data-baseweb="popover"] *, [data-baseweb="tooltip"] * {{ color: #fff !important; }}

  label[for="model_select_compact"] {{ font-size: 50px !important; font-weight: bold !important; }}
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child {{ font-size: 40px !important; }}
  div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] div[role="option"] {{ font-size: 35px !important; }}

  div.stButton > button {{ font-size: 35px !important; font-weight: bold !important; height: 50px !important; }}
  #action-row {{ display:flex; align-items:center; gap:10px; }}
  .stSelectbox, .stButton {{ font-size:35px !important; }}
</style>
""")

css("""
<style>
#leftwrap { position: relative; top: -80px; }
.block-container [data-testid="stHorizontalBlock"] > div:has(.form-banner) [data-testid="stHorizontalBlock"] {
  position: relative !important; top: -60px !important;
}
.prediction-result{ white-space: nowrap !important; display: inline-block !important; width: auto !important; line-height: 1.2 !important; margin-top: 0 !important; }
</style>
""")

with st.sidebar:
    right_offset = st.slider("Right panel vertical offset (px)", min_value=-200, max_value=1000, value=50, step=2)

# =============================================================================
# Step #3: Title + adjustable logo position and size (HEADER ONLY)
# =============================================================================
try:
    _logo_path = BASE_DIR / "TJU logo.png"
    _b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii") if _logo_path.exists() else ""
except Exception:
    _b64 = ""

with st.sidebar:
    st.markdown("### Header position (title & logo)")
    HEADER_X = st.number_input("Header X offset (px)", min_value=-2000, max_value=6000, value=0, step=20)
    TITLE_LEFT = st.number_input("Title X (px)", min_value=-1000, max_value=5000, value=180, step=10)
    TITLE_TOP  = st.number_input("Title Y (px)",  min_value=-500,  max_value=500,  value=40,  step=2)
    LOGO_LEFT  = st.number_input("Logo X (px)",   min_value=-1000, max_value=5000, value=340, step=10)
    LOGO_TOP   = st.number_input("Logo Y (px)",   min_value=-500,  max_value=500,  value=60,  step=2)
    LOGO_SIZE  = st.number_input("Logo size (px)", min_value=20, max_value=400, value=80, step=2)

st.markdown(f"""
<style>
  .page-header {{ display: flex; align-items: center; justify-content: flex-start; gap: 20px; margin: 0; padding: 0; }}
  .page-header__title {{ font-size: {FS_TITLE}px; font-weight: 800; margin: 0; transform: translate({int(TITLE_LEFT)}px, {int(TITLE_TOP)}px); }}
  .page-header__logo {{ height: {int(LOGO_SIZE)}px; width: auto; display: block; transform: translate({int(LOGO_LEFT)}px, {int(LOGO_TOP)}px); }}
</style>
<div class="page-header-outer" style="width:100%; transform: translateX({int(HEADER_X)}px) !important; will-change: transform;">
  <div class="page-header">
    <div class="page-header__title">Predict Damage index (DI) for RC Shear Walls</div>
    {f'<img class="page-header__logo" alt="Logo" src="data:image/png;base64,{_b64}" />' if _b64 else ''}
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
html, body{ margin:0 !important; padding:0 !important; }
header[data-testid="stHeader"]{ height:0 !important; padding:0 !important; background:transparent !important; }
header[data-testid="stHeader"] *{ display:none !important; }
div.stApp{ margin-top:-4rem !important; }
section.main > div.block-container{ padding-top:0 !important; margin-top:0 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    app_x = st.slider("Global horizontal offset (px)", min_value=0, max_value=1600, value=800, step=10)

st.markdown(f"""
<style>
:root {{ --shift-right: {int(app_x)}px; }}
[data-testid="stAppViewContainer"]{{ padding-left: var(--shift-right) !important; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Step #4: Model loading (robust; tolerates different names/paths)
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

# --- helper for ANN model loading (tries tf.keras first, then keras-3 if present) ---
def _load_any_keras_model(candidates):
    p = pfind(list(candidates))
    # TensorFlow loader
    try:
        m = load_model(p)
        return m, f"tf.keras ({p.name})"
    except Exception as e_tf:
        # Optional fallback for models saved with keras>=3
        try:
            import keras as k3
            m = k3.saving.load_model(p)
            return m, f"keras3 ({p.name})"
        except Exception as e_k3:
            raise RuntimeError(f"ANN load failed for {p.name}: tf.keras={e_tf} | keras3={e_k3}")

# --- helper for scaler loading (joblib first, optional skops fallback) ---
def _load_scaler_any(*cands):
    path = pfind(list(cands))
    try:
        obj = joblib.load(path)
        return obj, f"joblib ({path.name})"
    except Exception as e_job:
        try:
            import skops.io as sio  # only if installed
            obj = sio.load(path, trusted=True)
            return obj, f"skops ({path.name})"
        except Exception as e_sk:
            raise RuntimeError(f"Scaler load failed for {path.name}: joblib={e_job} | skops={e_sk}")

# -------------------- PS (ANN) --------------------
ann_ps_model = None; ann_ps_proc = None
try:
    ann_ps_model, src = _load_any_keras_model(["ANN_PS_Model.keras", "ANN_PS_Model.h5"])
    sx, x_kind = _load_scaler_any("ANN_PS_Scaler_X.save","ANN_PS_Scaler_X.pkl","ANN_PS_Scaler_X.joblib","ANN_PS_Scaler_X.skops")
    sy, y_kind = _load_scaler_any("ANN_PS_Scaler_y.save","ANN_PS_Scaler_y.pkl","ANN_PS_Scaler_y.joblib","ANN_PS_Scaler_y.skops")
    ann_ps_proc = _ScalerShim(sx, sy)
    ann_ps_proc.x_kind = x_kind; ann_ps_proc.y_kind = y_kind
    record_health("PS (ANN)", True, f"loaded via {src}")
except Exception as e:
    record_health("PS (ANN)", False, f"{e}")

# -------------------- MLP (ANN) -------------------
ann_mlp_model = None; ann_mlp_proc = None
try:
    ann_mlp_model, src = _load_any_keras_model(["ANN_MLP_Model.keras", "ANN_MLP_Model.h5"])
    sx, x_kind = _load_scaler_any("ANN_MLP_Scaler_X.save","ANN_MLP_Scaler_X.pkl","ANN_MLP_Scaler_X.joblib","ANN_MLP_Scaler_X.skops")
    sy, y_kind = _load_scaler_any("ANN_MLP_Scaler_y.save","ANN_MLP_Scaler_y.pkl","ANN_MLP_Scaler_y.joblib","ANN_MLP_Scaler_y.skops")
    ann_mlp_proc = _ScalerShim(sx, sy)
    ann_mlp_proc.x_kind = x_kind; ann_mlp_proc.y_kind = y_kind
    record_health("MLP (ANN)", True, f"loaded via {src}")
except Exception as e:
    record_health("MLP (ANN)", False, f"{e}")

# ----- Random Forest (joblib FIRST even for .json; fall back to SKOPS) -----
rf_model = None
try:
    rf_path = pfind([
        "random_forest_model.pkl", "random_forest_model.joblib",
        "rf_model.pkl", "RF_model.pkl",
        "Best_RF_Model.json", "best_rf_model.json", "RF_model.json"
    ])
    try:
        rf_model = joblib.load(rf_path)  # your .json is a joblib pickle
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
    record_health("CatBoost", False, str(e))

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

with st.sidebar:
    st.header("Model Health")
    for name, ok, msg, cls in health:
        st.markdown(f"- <span class='{cls}'>{'✅' if ok else '❌'} {name}</span><br/><small>{msg}</small>", unsafe_allow_html=True)
    for label, proc in [("PS scaler", ann_ps_proc), ("MLP scaler", ann_mlp_proc)]:
        try: st.caption(f"{label}: X={proc.x_kind} | Y={proc.y_kind}")
        except Exception: pass

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# =============================================================================
# Step #5: Ranges, inputs, layout
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
    (rf"$f_{{yl}}{U('MPa')}$",    "fyl",  400.0, 1.0, None, "Vertical web yield strength"),
    (rf"$f_{{ybl}}{U('MPa')}$",   "fybl", 400.0, 1.0, None, "Vertical boundary yield strength"),
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

left, right = st.columns([1.5, 2], gap="large")

with left:
    st.markdown("<div class='left-panel'>", unsafe_allow_html=True)

    st.markdown("<div class='form-banner'>Inputs Features</div>", unsafe_allow_html=True)

    st.markdown("<style>.section-header{margin:.2rem 0 !important;}</style>", unsafe_allow_html=True)

    css("<div id='leftwrap'>")
    css("<div id='compact-form'>")

    c1, _gap, c2 = st.columns([1, 0.08, 1], gap="large")

    with c1:
        st.markdown("<div class='section-header'>Geometry </div>", unsafe_allow_html=True)
        lw, hw, tw, b0, db, AR, M_Vlw = [num(*row) for row in GEOM]
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fc, fyt, fysh = [num(*row) for row in MATS[:3]]

    with c2:
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fyl, fybl = [num(*row) for row in MATS[3:]]
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Reinf. Ratios </div>", unsafe_allow_html=True)
        rt, rsh, rl, rbl, s_db, axial, theta = [num(*row) for row in REINF]

    css("</div>")
    css("</div>")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# Step #6: Right panel (unchanged)
# =============================================================================
HERO_X, HERO_Y, HERO_W = 100, 5, 550
MODEL_X, MODEL_Y = 100, -2
CHART_W = 550

with right:
    st.markdown(f"<div style='height:{int(right_offset)}px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="position:relative; left:{int(HERO_X)}px; top:{int(HERO_Y)}px; text-align:left;">
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
    [data-baseweb="popover"], [data-baseweb="popover"] > div { background: transparent !important; box-shadow: none !important; border: none !important; }
    div[data-testid="stSelectbox"] > div > div { height: 50px !important; display:flex !important; align-items:center !important; margin-top: -0px; }
    div[data-testid="stSelectbox"] label p { font-size: 18px !important; color: black !important; font-weight: bold !important; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child { font-size: 30px !important; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[role="listbox"] div[role="option"] { font-size: 30px !important; color: black !important; }
    [data-baseweb="select"] *, [data-baseweb="popover"] *, [data-baseweb="menu"] * { color: black !important; background-color: #D3D3D3 !important; font-size: 30px !important; }
    div[data-testid="stButton"] button p { font-size: 30px !important; color: black !important; font-weight: normal !important; }
    div[role="option"] { color: black !important; font-size: 16px !important; }
    div.stButton > button { height: 50px !important; display:flex; align-items:center; justify-content:center; }
    #action-row { display:flex; align-items:center; gap: 1px; }
    .stAltairChart { transform: translate(100px, 50px) !important; }
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
        st.markdown("<div id='three-btns' style='margin-top:35px;'>", unsafe_allow_html=True)
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
        dl_slot.download_button("📂 Download All Results as CSV", data=csv, file_name="di_predictions.csv", mime="text/csv", use_container_width=False, key="dl_csv_main")

    col1, col2 = st.columns([0.01, 20])
    with col2:
        chart_slot = st.empty()

# =============================================================================
# Step #7: Prediction utilities & curve helpers
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
    # keep training order
    df_trees = _df_in_train_order(input_df)

    # ensure finite values for tree models
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
                    theta_max: float = THETA_MAX, di_max: float = 1.5, size: int = 460):
    import altair as alt
    selection = alt.selection_point(name='select', fields=['θ', 'Predicted_DI'], nearest=True, on='mouseover', empty=False, clear='mouseout')
    AXIS_LABEL_FS = 20; AXIS_TITLE_FS = 24; TICK_SIZE = 8; TITLE_PAD = 12; LABEL_PAD = 8
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
    line_layer = alt.Chart(curve).mark_line(strokeWidth=3).encode(x="θ:Q", y="Predicted_DI:Q").properties(width=size, height=size)

    k = 3
    if not curve.empty:
        curve_points = curve.iloc[::k].copy()
        if not curve_points.empty and (curve_points.iloc[-1]["θ"] != curve.iloc[-1]["θ"]):
            curve_points = pd.concat([curve_points, curve.tail(1)], ignore_index=True)
    else:
        curve_points = pd.DataFrame({"θ": [], "Predicted_DI": []})

    points_layer = alt.Chart(curve_points).mark_circle(size=100, opacity=0.7).encode(
        x="θ:Q", y="Predicted_DI:Q",
        tooltip=[alt.Tooltip("θ:Q", title="Drift Ratio (θ)", format=".2f"),
                 alt.Tooltip("Predicted_DI:Q", title="Predicted DI", format=".4f")]
    ).add_params(selection)

    rules_layer = alt.Chart(curve).mark_rule(color='red', strokeWidth=2).encode(x="θ:Q", y="Predicted_DI:Q").transform_filter(selection)
    text_layer = alt.Chart(curve).mark_text(align='left', dx=10, dy=-10, fontSize=20, fontWeight='bold', color='red').encode(
        x="θ:Q", y="Predicted_DI:Q", text=alt.Text("Predicted_DI:Q", format=".4f")
    ).transform_filter(selection)

    chart = (alt.layer(axes_layer, line_layer, points_layer, rules_layer, text_layer)
             .configure_view(strokeWidth=0)
             .configure_axis(domain=True, ticks=True)
             .configure(padding={"left": 6, "right": 6, "top": 6, "bottom": 6}))
    chart_html = chart.to_html()
    chart_html = chart_html.replace('</style>',
        '</style><style>.vega-embed .vega-tooltip, .vega-embed .vega-tooltip * { font-size: 32px !important; font-weight: bold !important; background: #000 !important; color: #fff !important; padding: 20px !important; }</style>')
    st.components.v1.html(chart_html, height=size + 100)

# =============================================================================
# Step #8: Predict on click; always render curve
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
            dl_slot.download_button("📂 Download All Results as CSV", data=csv, file_name="di_predictions.csv",
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
# Step #9: Optional "Recent Predictions" (hidden by default)
# =============================================================================
show_recent = st.sidebar.checkbox("Show Recent Predictions", value=False)
if show_recent and not st.session_state.results_df.empty:
    right_predictions = st.empty()
    with right_predictions:
        st.markdown("### 🧾 Recent Predictions")
        for i, row in st.session_state.results_df.tail(5).reset_index(drop=True).iterrows():
            st.markdown(
                f"<div class='recent-box' style='display:inline-block; width:auto; padding:4px 10px;'>"
                f"Pred {i+1} ➔ DI = {row['Predicted_DI']:.4f}</div>",
                unsafe_allow_html=True
            )
