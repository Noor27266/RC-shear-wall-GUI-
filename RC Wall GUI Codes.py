DOC_NOTES = """
RC Shear Wall Damage Index (DI) Estimator — compact, same logic/UI
"""


# =============================================================================
# 🚀 STEP 1: CORE IMPORTS & TENSORFLOW BACKEND SETUP
# =============================================================================

# =============================================================================
# 🚀 SUB STEP 1.1: ENVIRONMENT CONFIGURATION
# =============================================================================
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# =============================================================================
# 🚀 SUB STEP 1.2: CORE LIBRARY IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path
from glob import glob

# =============================================================================
# 🚀 SUB STEP 1.3: MACHINE LEARNING LIBRARY IMPORTS
# =============================================================================
# ML libs
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb

# =============================================================================
# 🚀 SUB STEP 1.4: KERAS COMPATIBILITY LOADER SETUP
# =============================================================================
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


# =============================================================================
# 🚀 SUB STEP 1.5: SESSION STATE INITIALIZATION
# =============================================================================
# --- session defaults (prevents AttributeError on first run) ---
st.session_state.setdefault("results_df", pd.DataFrame())

# =============================================================================
# 🔧 STEP 2: UTILITY FUNCTIONS & HELPER TOOLS
# =============================================================================

css = lambda s: st.markdown(s, unsafe_allow_html=True)


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def dv(R, key, proposed):
    lo, hi = R[key]
    return float(max(lo, min(proposed, hi)))


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

st.set_page_config(
    page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱"
)

# Header / spacing
st.markdown(
    """
<style>
html, body{
    margin:0 !important;
    padding:0 !important;
    overflow:hidden !important;        /* <- added to hide page scrollbars */
}
header[data-testid="stHeader"]{ height:0 !important; padding:0 !important; background:transparent !important; }
header[data-testid="stHeader"] *{ display:none !important; }



/* Remove extra white space at top */
section.main > div.block-container{
    padding-top:0 !important;
    margin-top:-2.5rem !important;  /* Reduced from -2.0rem to -1.5rem */
}

/* Keep Altair responsive */
.vega-embed, .vega-embed .chart-wrapper{
    max-width:100% !important;
}

/* (old scrollbar-forcing code removed) */
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 🎨 SUB STEP 3.1: FONT SIZE SCALING CONFIGURATION
# =============================================================================
SCALE_UI = 0.36

s = lambda v: int(round(v * SCALE_UI))

FS_TITLE = s(20)
FS_SECTION = s(60)
FS_LABEL = s(50)
FS_UNITS = s(30)
FS_INPUT = s(30)
FS_SELECT = s(35)
FS_BUTTON = s(20)
FS_BADGE = s(30)
FS_RECENT = s(20)
INPUT_H = max(32, int(FS_INPUT * 2.0))

# =============================================================================
# 🎨 SUB STEP 3.2: COLOR SCHEME DEFINITION
# =============================================================================
DEFAULT_LOGO_H = 45
PRIMARY = "#8E44AD"
SECONDARY = "#f9f9f9"
INPUT_BG = "#ffffff"
INPUT_BORDER = "#e6e9f2"
LEFT_BG = "#e0e4ec"

# =============================================================================
# 🎨 STEP 3.3: COMPREHENSIVE CSS STYLING & THEME SETUP
# =============================================================================
css(
    f"""
<style>
  .block-container {{
    padding-top: 1.5rem !important;  /* Reduced from 2.5rem to 1.5rem */
    padding-bottom: 0.5rem !important;
    max-height: none !important;
    overflow: visible !important;
}}

  h1 {{
      font-size:{FS_TITLE}px !important;
      margin:0 rem 0 !important;
  }}

  .section-header {{
      font-size:{FS_SECTION}px !important;
      font-weight:700;
      margin:.35rem 0;
  }}

  .stNumberInput label,
  .stSelectbox label {{
      font-size:{FS_LABEL}px !important;
      font-weight:700;
  }}

  .stNumberInput label .katex,
  .stSelectbox label .katex {{
      font-size:{FS_LABEL}px !important;
      line-height:1.2 !important;
  }}

  .stNumberInput label .katex .fontsize-ensurer,
  .stSelectbox label .katex .fontsize-ensurer {{
      font-size:1em !important;
  }}

  .stNumberInput label .katex .mathrm,
  .stSelectbox label .katex .mathrm {{
      font-size:{FS_UNITS}px !important;
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
      box-shadow:0 1px 2px rgba(16,24,40,.06) !important;
      transition:border-color .15s ease, box-shadow .15s ease !important;
  }}

  div[data-testid="stNumberInput"] button {{
      background:#ffffff !important;
      border:1px solid {INPUT_BORDER} !important;
      border-radius:10px !important;
      box-shadow:0 1px 1px rgba(16,24,40,.05) !important;
  }}

  .stSelectbox [role="combobox"],
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child,
  div[data-testid="stSelectbox"] div[role="listbox"],
  div[data-testid="stSelectbox"] div[role="option"] {{
      font-size:{FS_SELECT}px !important;
  }}

  div.stButton > button {{
      font-size:{FS_BUTTON}px !important;
      height:{max(42, int(round(FS_BUTTON*1.45)))}px !important;
      line-height:{max(36, int(round(FS_BUTTON*1.15)))}px !important;
      color:#fff !important;
      font-weight:700;
      border:none !important;
      border-radius:8px !important;
      background:#4CAF50 !important;
  }}

  button[key="reset_btn"] {{
      background:#2196F3 !important;
  }}

  button[key="clear_btn"] {{
      background:#f44336 !important;
  }}

  #compact-form {{
      max-width:900px;
      margin:0 auto;
  }}

  html, body, #root, .stApp, section.main, .block-container, [data-testid="stAppViewContainer"] {{
      background: linear-gradient(90deg, #e0e4ec 60%, transparent 60%) !important;
      min-height: 100vh !important;
      height: auto !important;
      overflow:hidden !important;      /* <- added so inner container cannot scroll */
  }}

  [data-testid="column"]:first-child {{
      min-height: 100vh !important;
      background: #e0e4ec !important;
      margin-top: 0px !important;  /* ADD THIS LINE - pushes left column down */
      padding-top: 0px !important;  /* ADD THIS LINE - adds space at top */
  }}
</style>
"""
)



# =============================================================================
# 🏷️ LOGO LOADING FOR LEFT PANEL
# =============================================================================
try:
    _logo_path = BASE_DIR / "TJU logo.png"
    _b64 = (
        base64.b64encode(_logo_path.read_bytes()).decode("ascii")
        if _logo_path.exists()
        else ""
    )
except Exception:
    _b64 = ""



# =============================================================================
# 🤖 STEP 5: MACHINE LEARNING MODEL LOADING & HEALTH CHECKING
# =============================================================================

def record_health(name, ok, msg=""):
    """Keep a log of which models loaded successfully."""
    health.append((name, ok, msg, "ok" if ok else "err"))

health = []

class _ScalerShim:
    """Wrapper to keep X / y scalers together for ANN models."""
    def __init__(self, X_scaler, Y_scaler):
        import numpy as _np
        self._np = _np
        self.Xs = X_scaler
        self.Ys = Y_scaler
        self.x_kind = "External joblib"
        self.y_kind = "External joblib"

    def transform_X(self, X):
        return self.Xs.transform(X)

    def inverse_transform_y(self, y):
        y = self._np.array(y).reshape(-1, 1)
        return self.Ys.inverse_transform(y)

# ---------------------------- PS (ANN) ---------------------------------------
ann_ps_model = None
ann_ps_proc  = None
try:
    ps_model_path = pfind(["ANN_PS_Model.keras", "ANN_PS_Model.h5"])
    ann_ps_model  = _load_keras_model(ps_model_path)

    sx = joblib.load(
        pfind([
            "ANN_PS_Scaler_X.save",
            "ANN_PS_Scaler_X.pkl",
            "ANN_PS_Scaler_X.joblib",
        ])
    )
    sy = joblib.load(
        pfind([
            "ANN_PS_Scaler_y.save",
            "ANN_PS_Scaler_y.pkl",
            "ANN_PS_Scaler_y.joblib",
        ])
    )
    ann_ps_proc = _ScalerShim(sx, sy)
    record_health("PS (ANN)", True, f"loaded from {ps_model_path}")
except Exception as e:
    record_health("PS (ANN)", False, f"{e}")

# ---------------------------- MLP (ANN) --------------------------------------
ann_mlp_model = None
ann_mlp_proc  = None
try:
    mlp_model_path = pfind(["ANN_MLP_Model.keras", "ANN_MLP_Model.h5"])
    ann_mlp_model  = _load_keras_model(mlp_model_path)

    sx = joblib.load(
        pfind([
            "ANN_MLP_Scaler_X.save",
            "ANN_MLP_Scaler_X.pkl",
            "ANN_MLP_Scaler_X.joblib",
        ])
    )
    sy = joblib.load(
        pfind([
            "ANN_MLP_Scaler_y.save",
            "ANN_MLP_Scaler_y.pkl",
            "ANN_MLP_Scaler_y.joblib",
        ])
    )
    ann_mlp_proc = _ScalerShim(sx, sy)
    record_health("MLP (ANN)", True, f"loaded from {mlp_model_path}")
except Exception as e:
    record_health("MLP (ANN)", False, f"{e}")

# ---------------------------- Random Forest ----------------------------------
rf_model = None
try:
    rf_path = pfind([
        "random_forest_model.pkl",
        "random_forest_model.joblib",
        "rf_model.pkl",
        "RF_model.pkl",
        "Best_RF_Model.json",
        "best_rf_model.json",
        "RF_model.json",
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
            record_health(
                "Random Forest",
                False,
                f"RF load failed for {rf_path} (joblib: {e_joblib}) (skops: {e_skops})",
            )
except Exception as e:
    record_health("Random Forest", False, str(e))

# ---------------------------- XGBoost ----------------------------------------
xgb_model = None
try:
    xgb_path = pfind([
        "XGBoost_trained_model_for_DI.json",
        "Best_XGBoost_Model.json",
        "xgboost_model.json",
    ])
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    record_health("XGBoost", True, f"loaded from {xgb_path}")
except Exception as e:
    record_health("XGBoost", False, str(e))

# ---------------------------- CatBoost ---------------------------------------
cat_model = None
try:
    cat_path = pfind([
        "CatBoost.cbm",
        "Best_CatBoost_Model.cbm",
        "catboost.cbm",
    ])
    cat_model = catboost.CatBoostRegressor()
    cat_model.load_model(cat_path)
    record_health("CatBoost", True, f"loaded from {cat_path}")
except Exception as e:
    record_health("CatBoost", False, f"{e}")

# ---------------------------- LightGBM ---------------------------------------
def load_lightgbm_flex():
    try:
        p = pfind([
            "LightGBM_model.txt",
            "Best_LightGBM_Model.txt",
            "LightGBM_model.bin",
            "LightGBM_model.pkl",
            "LightGBM_model.joblib",
            "LightGBM_model",
        ])
    except Exception:
        raise FileNotFoundError("No LightGBM model file found.")

    try:
        return lgb.Booster(model_file=str(p)), "booster", p
    except Exception:
        try:
            return joblib.load(p), "sklearn", p
        except Exception as e:
            raise e

try:
    lgb_model, lgb_kind, lgb_path = load_lightgbm_flex()
    record_health("LightGBM", True, f"loaded as {lgb_kind} from {lgb_path}")
except Exception as e:
    lgb_model = None
    record_health("LightGBM", False, str(e))

# ---------------------------- Registry ---------------------------------------
model_registry = {}

for name, ok, *_ in health:
    if not ok:
        continue
    if name == "XGBoost" and xgb_model is not None:
        model_registry["XGBoost"] = xgb_model
    elif name == "LightGBM" and lgb_model is not None:
        model_registry["LightGBM"] = lgb_model
    elif name == "CatBoost" and cat_model is not None:
        model_registry["CatBoost"] = cat_model
    elif name == "PS (ANN)" and ann_ps_model is not None:
        model_registry["PS"] = ann_ps_model
    elif name == "MLP (ANN)" and ann_mlp_model is not None:
        model_registry["MLP"] = ann_mlp_model
    elif name == "Random Forest" and rf_model is not None:
        model_registry["Random Forest"] = rf_model

# global model ordering + label mapping (used in UI + prediction)
MODEL_ORDER = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
LABEL_TO_KEY = {"RF": "Random Forest"}

# =============================================================================
# 📊 STEP 6: INPUT PARAMETERS & DATA RANGES DEFINITION
# =============================================================================
R = {
    "lw": (400.0, 3500.0),
    "hw": (495.0, 5486.4),
    "tw": (26.0, 305.0),
    "fc": (13.38, 93.6),
    "fyt": (0.0, 1187.0),
    "fysh": (0.0, 1375.0),
    "fyl": (160.0, 1000.0),
    "fybl": (0.0, 900.0),
    "rt": (0.000545, 0.025139),
    "rsh": (0.0, 0.041888),
    "rl": (0.0, 0.029089),
    "rbl": (0.0, 0.031438),
    "axial": (0.0, 0.86),
    "b0": (45.0, 3045.0),
    "db": (0.0, 500.0),
    "s_db": (0.0, 47.65625),
    "AR": (0.388889, 5.833333),
    "M_Vlw": (0.388889, 4.1),
    "theta": (0.0275, 4.85),
}
THETA_MAX = R["theta"][1]
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


def num(label, key, default, step, fmt, help_):
    return st.number_input(
        label,
        value=dv(R, key, default),
        step=step,
        min_value=R[key][0],
        max_value=R[key][1],
        format=fmt if fmt else None,
        help=help_,
    )


# Hide +/- buttons
css(
    """
<style>
div[data-testid="stNumberInput"] button {
    display: none !important;
}
</style>
"""
)

# =============================================================================
# 📊 SUB STEP 6.1: LAYOUT COLUMNS SETUP
# =============================================================================
left, right = st.columns([1.5, 1], gap="large")

# =============================================================================
# 📊 SUB STEP 6.2: LEFT PANEL CONTENT IMPLEMENTATION
# =============================================================================
with left:
    # LOGO CENTERED AT TOP - Bigger and moved up
    if _b64:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2px;">
            <img src="data:image/png;base64,{_b64}" 
                 style="height: 90px; width: auto;" 
                 alt="Logo" />
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 0px; margin: 0; padding: 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:transparent; border-radius:12px; padding:0px; margin:0 0 3px 0; box-shadow:none;">
        <div style="text-align:center; font-size:25px; font-weight:600; color:#333; margin:0; padding:0;">
            Predict Damage index for RC Shear Walls
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin: 0 0 0 0; padding: 0;">
        <div class='form-banner'>Inputs Features</div>
    """, unsafe_allow_html=True)

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

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# 🎮 STEP 7: RIGHT PANEL - CONTROLS & INTERACTION ELEMENTS
# =============================================================================
# Fixed-height box for schematic so DI–θ plot position does not change
SCHEM_BOX_H    = 300   # total vertical space reserved for schematic (keep this fixed)
SCHEM_IMG_H    = 450   # actual schematic image height
SCHEM_OFFSET_X = 80    # move schematic right (+) / left (-)
SCHEM_OFFSET_Y = -40   # move schematic down (+) / up (-)

SCHEM2_IMG_H   = 470   # SAME height for second schematic
SCHEM2_OFFSET_X = 500  # Just 40px more than original 380 - slight move to the right

CHART_W = 350          # width used later for DI–θ chart

with right:

    # --- TWO schematics side by side in fixed-height box ---
    st.markdown(
        f"""
        <div style="position:relative; height:{SCHEM_BOX_H}px; margin-bottom:0;">
            <!-- First schematic -->
            <img src="data:image/png;base64,{b64(BASE_DIR / "logo2-01.png")}"
                 style="
                    position:absolute;
                    left:{SCHEM_OFFSET_X}px;
                    top:{SCHEM_OFFSET_Y}px;
                    height:{SCHEM_IMG_H}px;
                    width:auto;
                 " />
            <!-- Second schematic -->
            <img src="data:image/png;base64,{b64(BASE_DIR / "RC shear wall schematic2.png")}"
                 style="
                    position:absolute;
                    left:{SCHEM2_OFFSET_X}px;
                    top:{SCHEM_OFFSET_Y}px;
                    height:{SCHEM2_IMG_H}px;
                    width:auto;
                 " />
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ---- ONE ROW: [ left = DI–θ plot | right = controls ] ----
    col_plot, col_controls = st.columns([3, 1])
    # =============================================================================
    # ⭐ SUB-STEP 7.1 — DI–θ PLOT (LEFT SIDE)
    # =============================================================================
    with col_plot:
        # slot where STEP 11 will render the DI–θ plot
        chart_slot = st.empty()
           # =============================================================================
    # ⭐ SUB-STEP 7.2 — MODEL SELECTION + BUTTONS (RIGHT SIDE)
    # =============================================================================
    with col_controls:
        
        # REMOVE THIS LINE: st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        
        # Model selection
        available = set(model_registry.keys())
        ordered_keys = [m for m in MODEL_ORDER if m in available] or ["(no models loaded)"]
        display_labels = ["RF" if m == "Random Forest" else m for m in ordered_keys]

        model_choice_label = st.selectbox(
            "Model Selection",
            display_labels,
            key="model_select_compact",
        )
        model_choice = LABEL_TO_KEY.get(model_choice_label, model_choice_label)

        # Buttons - USE UNIQUE KEYS
        submit = st.button("Calculate", key=f"calc_btn_{st.session_state.get('calc_counter', 0)}", use_container_width=True)

        if st.button("Reset", key="reset_btn_main", use_container_width=True):
            st.rerun()

        if st.button("Clear All", key="clear_btn_main", use_container_width=True):
            st.session_state.results_df = pd.DataFrame()

        # Latest DI + CSV download
        if not st.session_state.results_df.empty:
            latest_pred = st.session_state.results_df.iloc[-1]["Predicted_DI"]
            st.markdown(
                f"<div class='prediction-with-color'>Predicted Damage Index : {latest_pred:.4f}</div>",
                unsafe_allow_html=True,
            )

            csv = st.session_state.results_df.to_csv(index=False)
            st.download_button(
                "📂 Download as CSV",
                data=csv,
                file_name="di_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_csv_main",
            )

        # ADD THESE 3 LINES RIGHT HERE (inside col_controls)
        if submit:
            st.session_state["calc_counter"] = st.session_state.get("calc_counter", 0) + 1
            st.rerun()


    # styling for the blue DI label (unchanged)
    st.markdown(
        f"""
    <style>
    .prediction-with-color {{
        color: #2e86ab !important;
        font-weight: 700 !important;
        font-size: {FS_BADGE}px !important;
        background: #f1f3f4 !important;
        padding: 10px 12px !important;
        border-radius: 6px !important;
        text-align: center !important;
        margin: 0 !important;
        height: 45px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 180px !important;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

css("""
<style>
/* Adjust positioning for right panel elements */
div[data-testid="stSelectbox"],
div.stButton,
div[data-testid="stDownloadButton"],
.prediction-with-color {
    position: relative !important;
    top: 140px !important;  /* Changed from 200px to 10px */
    left: 20px !important;
    margin-bottom: 8px !important;  /* Reduced from 15px to 8px for tighter spacing */
}

/* Also adjust the chart container to move it up */
div[data-testid="column"]:nth-child(2) {
    margin-top: -20px !important;  /* Pull the entire right column up */
}
</style>
""")


# =============================================================================
# ⚡ STEP 8: DI–θ PREDICTION & PLOT (ALL CODE HERE)
# =============================================================================

_TRAIN_NAME_MAP = {
    "l_w": "lw",
    "h_w": "hw",
    "t_w": "tw",
    "f′c": "fc",
    "fyt": "fyt",
    "fysh": "fysh",
    "fyl": "fyl",
    "fybl": "fybl",
    "ρt": "pt",
    "ρsh": "psh",
    "ρl": "pl",
    "ρbl": "pbl",
    "P/(Agf′c)": "P/(Agfc)",
    "b0": "b0",
    "db": "db",
    "s/db": "s/db",
    "AR": "AR",
    "M/Vlw": "M/Vlw",
    "θ": "θ",
}

_TRAIN_COL_ORDER = [
    "lw","hw","tw","fc","fyt","fysh","fyl","fybl",
    "pt","psh","pl","pbl","P/(Agfc)","b0","db","s/db",
    "AR","M/Vlw","θ",
]

def _df_in_train_order(df): 
    return df.rename(columns=_TRAIN_NAME_MAP).reindex(columns=_TRAIN_COL_ORDER)


def predict_di(choice, _unused_array, input_df):
    df_trees = _df_in_train_order(input_df).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    X = df_trees.values.astype(np.float32)

    if choice == "LightGBM":
        prediction = float(model_registry["LightGBM"].predict(X)[0])
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
        except:
            model_registry["PS"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["PS"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_ps_proc.inverse_transform_y(yhat).item())

    if choice == "MLP":
        Xn = ann_mlp_proc.transform_X(X)
        try:
            yhat = model_registry["MLP"].predict(Xn, verbose=0)[0][0]
        except:
            model_registry["MLP"].compile(optimizer="adam", loss="mse")
            yhat = model_registry["MLP"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_mlp_proc.inverse_transform_y(yhat).item())

    return max(0.035, min(prediction, 1.5))


def _make_input_df(lw,hw,tw,fc,fyt,fysh,fyl,fybl,rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw,theta):
    cols = ["l_w","h_w","t_w","f′c","fyt","fysh","fyl","fybl","ρt","ρsh","ρl","ρbl",
            "P/(Agf′c)","b0","db","s/db","AR","M/Vlw","θ"]
    vals = [lw,hw,tw,fc,fyt,fysh,fyl,fybl,rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw,theta]
    return pd.DataFrame([vals], columns=cols)


def _sweep_curve_df(model_choice, base_df, theta_max=THETA_MAX, step=0.10):
    actual_theta = float(base_df.iloc[0]["θ"])
    thetas = np.round(np.arange(0, actual_theta+1e-9, step), 2)

    rows=[]
    for th in thetas:
        df = base_df.copy()
        df["θ"] = th
        di = predict_di(model_choice, None, df)
        rows.append({"θ":th, "Predicted_DI":di})

    return pd.DataFrame(rows)


def _damage_state_label(di):
    if di < 0.2: return "Undamage"
    if di < 0.5: return "Partial Damage"
    if di <= 1.0: return "Severe Damage"
    return "Collapse"


def render_di_chart(curve_df, highlight_df=None, theta_max=THETA_MAX, di_max=1.5, size=460):
    import altair as alt

    if curve_df.empty:
        return

    # extend curve with last predicted point to remove the gap
    if highlight_df is not None:
        curve_df = pd.concat([curve_df, highlight_df], ignore_index=True)

    actual_theta_max = curve_df["θ"].max()

    AXIS_LABEL_FS = 14
    AXIS_TITLE_FS = 16

    base_axes_df = pd.DataFrame({"θ":[0, actual_theta_max], "Predicted_DI":[0,0]})
    x_ticks = np.linspace(0, actual_theta_max, 5).round(2)

    # ---- background bands (UD, PD, SD, COL ranges) ----
    bands_df = pd.DataFrame([
        {"y0":0.0, "y1":0.2, "color":"rgba(0,200,0,0.18)"},     # UD
        {"y0":0.2, "y1":0.5, "color":"rgba(255,215,0,0.18)"},   # PD
        {"y0":0.5, "y1":1.0, "color":"rgba(255,140,0,0.18)"},   # SD
        {"y0":1.0, "y1":1.5, "color":"rgba(255,0,0,0.18)"},     # COL
    ])

    band_layer = (
        alt.Chart(bands_df)
        .mark_rect()
        .encode(
            x=alt.value(0),
            x2=alt.value(size),
            y="y0:Q",
            y2="y1:Q",
            color=alt.Color("color:N", scale=None)
        )
        .properties(width=size, height=size)
    )

    # ---- main axes with tight y-limit at 1.5 ----
    axes_layer = (
        alt.Chart(base_axes_df).mark_line(opacity=0)
        .encode(
            x=alt.X(
                "θ:Q",
                title="Drift Ratio (θ)",
                scale=alt.Scale(domain=[0, actual_theta_max]),
                axis=alt.Axis(
                    values=list(x_ticks),
                    format=".2f",
                    labelFontSize=AXIS_LABEL_FS,
                    titleFontSize=AXIS_TITLE_FS,
                ),
            ),
            y=alt.Y(
                "Predicted_DI:Q",
                title="Damage Index (DI)",
                scale=alt.Scale(domain=[0, di_max], nice=False, clamp=True),
                axis=alt.Axis(
                    values=[0, 0.2, 0.5, 1.0, 1.5],
                    labelFontSize=AXIS_LABEL_FS,
                    titleFontSize=AXIS_TITLE_FS,
                ),
            ),
        )
        .properties(width=size, height=size)
    )

    line_layer = (
        alt.Chart(curve_df)
        .mark_line(strokeWidth=2)
        .encode(x="θ:Q", y="Predicted_DI:Q")
    )

    layers = [band_layer, axes_layer, line_layer]

    # ---- band labels: UD / PD / SD / COL ----
    labels_df = pd.DataFrame([
        {"θ": actual_theta_max * 0.80, "Predicted_DI": 0.10, "label": "UD"},
        {"θ": actual_theta_max * 0.80, "Predicted_DI": 0.35, "label": "PD"},
        {"θ": actual_theta_max * 0.20, "Predicted_DI": 0.75, "label": "SD"},
        {"θ": actual_theta_max * 0.20, "Predicted_DI": 1.25, "label": "COL"},
    ])

    label_layer = (
        alt.Chart(labels_df)
        .mark_text(
            fontSize=18,
            fontWeight="bold",
            color="black",
        )
        .encode(
            x="θ:Q",
            y="Predicted_DI:Q",
            text="label:N",
        )
    )

    layers.append(label_layer)

    # ---- highlight last prediction point + DI value ----
    if highlight_df is not None:
        point_layer = (
            alt.Chart(highlight_df)
            .mark_circle(size=110, color="blue")
            .encode(x="θ:Q", y="Predicted_DI:Q")
        )

        di_text_layer = (
            alt.Chart(highlight_df)
            .mark_text(
                align="center",
                dx=0,
                dy=18,
                fontSize=16,
                fontWeight="bold",
                color="red",
            )
            .encode(
                x="θ:Q",
                y="Predicted_DI:Q",
                text=alt.Text("Predicted_DI:Q", format=".4f"),
            )
        )

        layers += [point_layer, di_text_layer]

    chart = alt.layer(*layers).configure_view(strokeWidth=0)
    st.components.v1.html(chart.to_html(), height=size+100)


def _pick_default_model():
    for m in MODEL_ORDER:
        if m in model_registry:
            return m
    return None


# ---------------- MAIN EXECUTION ----------------

if "model_choice" not in locals():
    lbl = st.session_state.get("model_select_compact") or st.session_state.get("model_select")
    model_choice = LABEL_TO_KEY.get(lbl, lbl) if lbl else _pick_default_model()

if model_choice not in model_registry:
    st.error("No trained model available.")
else:
    # Submit is handled in STEP 7.2 - just update results here
    # Get the input values (they're in global scope)
    xdf = _make_input_df(
        lw,hw,tw,fc,fyt,fysh,fyl,fybl,
        rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw,theta
    )
    
    # Always calculate and update when we have a model choice
    try:
        pred = predict_di(model_choice, None, xdf)
        row = xdf.copy()
        row["Predicted_DI"] = pred
        # Only add if not already in results
        current_input = tuple(xdf.iloc[0].values)
        exists = False
        for _, r in st.session_state.results_df.iterrows():
            if tuple(r.drop('Predicted_DI', errors='ignore').values) == current_input:
                exists = True
                break
        if not exists and pred > 0:
            st.session_state.results_df = pd.concat(
                [st.session_state.results_df, row], ignore_index=True
            )
    except Exception as e:
        # Silent fail - just don't add to results
        pass

    if not st.session_state.results_df.empty:
        last = st.session_state.results_df.iloc[-1]
        last_di = float(last["Predicted_DI"])

        base = _make_input_df(
            lw,hw,tw,fc,fyt,fysh,fyl,fybl,
            rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw,
            float(last["θ"])
        )
        curve = _sweep_curve_df(model_choice, base, THETA_MAX, 0.1)

        highlight_df = pd.DataFrame({
            "θ": [float(last["θ"])],
            "Predicted_DI": [last_di],
        })

        with chart_slot.container():
            st.markdown("<div style='margin-top:150px;'>", unsafe_allow_html=True)
            render_di_chart(curve, highlight_df, THETA_MAX, 1.5, CHART_W)
            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# 🎨 STEP 9: FINAL UI POLISH & BANNER STYLING
# =============================================================================
st.markdown(
    """
<style>
.form-banner{
  background: linear-gradient(90deg, #0E9F6E, #84CC16) !important;
  color: #fff !important;
  text-align: center !important;
  border-radius: 10px !important;
  padding: .25rem .55rem !important;  /* Reduced further */
  margin-top: 2px !important;  /* Reduced from 5px to 2px */
  margin-bottom: 10px !important;  /* Reduced from 15px to 10px */
  transform: translateY(0) !important;
}
</style>
""",
    unsafe_allow_html=True,
)












