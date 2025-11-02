# -*- coding: utf-8 -*-

"""
RC Shear Wall Damage Index (DI) Estimator — clean & responsive Streamlit UI.
- No global "shift right"
- No fixed-width columns (responsive layout)
- No absolute transforms on big blocks
- Moderate font sizes to avoid wrapping
"""

# =============================================================================
# Imports
# =============================================================================
import os
# Do NOT force standalone keras backend; we use tf.keras bundled with TF.
# os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

# ML libs
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb

from tensorflow.keras.models import load_model  # use tf.keras only

# =============================================================================
# Small helpers
# =============================================================================
def css(s: str): st.markdown(s, unsafe_allow_html=True)
def b64(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""

def dv(R, key, proposed):
    lo, hi = R[key]
    return float(max(lo, min(proposed, hi)))

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

# --- Typography & sizing (keep reasonable to prevent wrapping)
FS_TITLE   = 42
FS_SECTION = 26
FS_LABEL   = 20
FS_UNITS   = 14
FS_INPUT   = 18
FS_SELECT  = 20
FS_BUTTON  = 20
FS_BADGE   = 18
FS_RECENT  = 14
INPUT_H    = max(36, int(FS_INPUT * 2.0))

PRIMARY   = "#8E44AD"
INPUT_BG     = "#ffffff"
INPUT_BORDER = "#e6e9f2"

# =============================================================================
# Global CSS (no absolute shifts; responsive-friendly)
# =============================================================================
css(f"""
<style>
  .block-container {{ padding-top: .5rem; }}

  h1.app-title {{
    font-size:{FS_TITLE}px !important;
    font-weight:800;
    margin: .3rem 0 1.0rem 0 !important;
    text-align:center;
  }}

  .section-header {{
    font-size:{FS_SECTION}px !important;
    font-weight:700;
    margin:.25rem 0 .5rem 0;
  }}

  /* labels (plain + math) */
  .stNumberInput label, .stSelectbox label {{
    font-size:{FS_LABEL}px !important; font-weight:700;
  }}
  .stNumberInput label .katex, .stSelectbox label .katex {{
    font-size:{FS_LABEL}px !important; line-height:1.2 !important;
  }}
  .stNumberInput label .katex .fontsize-ensurer,
  .stSelectbox label .katex .fontsize-ensurer {{ font-size:1em !important; }}

  /* units only */
  .stNumberInput label .katex .mathrm,
  .stSelectbox  label .katex .mathrm {{ font-size:{FS_UNITS}px !important; }}

  /* inputs */
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
      border-radius:10px !important;
      box-shadow:0 1px 2px rgba(16,24,40,.06) !important;
  }}

  /* buttons */
  div.stButton > button {{
    font-size:{FS_BUTTON}px !important;
    height:44px !important;
    color:#fff !important;
    font-weight:700;
    border:none !important;
    border-radius:8px !important;
    background:#4CAF50 !important;
    white-space:nowrap !important;
  }}
  div.stButton > button:hover {{ filter:brightness(0.95); }}

  /* specific button colors via data-key */
  button[key="calc_btn"] {{ background:#4CAF50 !important; }}
  button[key="reset_btn"] {{ background:#2196F3 !important; }}
  button[key="clear_btn"] {{ background:#f44336 !important; }}

  /* banner + badges */
  .form-banner {{
    text-align:center;
    background: linear-gradient(90deg, #0E9F6E, #84CC16);
    color:#fff; padding:.35rem .75rem; border-radius:10px;
    font-weight:800; font-size:{FS_SECTION + 2}px; margin:.2rem 0 .6rem 0 !important;
  }}
  .prediction-result {{
    font-size:{FS_BADGE}px !important; font-weight:700; color:#2e86ab;
    background:#f1f3f4; padding:.5rem .75rem; border-radius:8px;
    text-align:center; display:inline-block; white-space:nowrap;
  }}
  .recent-box {{
    font-size:{FS_RECENT}px !important; background:#f8f9fa; padding:.35rem .5rem;
    margin:.2rem .3rem .2rem 0; border-radius:6px;
    border-left:4px solid #4CAF50; font-weight:600; display:inline-block;
  }}

  /* Selectbox sizing */
  div[data-testid="stSelectbox"] label p {{ font-size:{FS_LABEL}px !important; font-weight:700 !important; }}
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child {{ font-size:{FS_SELECT}px !important; }}
</style>
""")

# =============================================================================
# Sidebar: small controls (no global shifts)
# =============================================================================
with st.sidebar:
    st.markdown("### Optional right panel offset")
    right_offset = st.slider(
        "Right panel vertical offset (px)",
        min_value=-100, max_value=400, value=30, step=2,
        help="Small vertical nudge for the right column"
    )

    # Header/Logo tweaks (moderate; no large transforms)
    st.markdown("---")
    st.markdown("### Header options")
    show_logo = st.checkbox("Show logo", value=True)
    logo_size = st.slider("Logo size (px)", 32, 200, 72, 2)

# =============================================================================
# Title + logo (inline, responsive)
# =============================================================================
logo_html = ""
if show_logo:
    logo_path = Path(__file__).resolve().parent / "TJU logo.png"
    _b64 = b64(logo_path)
    if _b64:
        logo_html = f'<img alt="Logo" src="data:image/png;base64,{_b64}" style="height:{int(logo_size)}px;vertical-align:middle;margin-left:.5rem"/>'

st.markdown(f'<h1 class="app-title">Predict Damage index (DI) for RC Shear Walls {logo_html}</h1>',
            unsafe_allow_html=True)

# =============================================================================
# Model loading (tolerant)
# =============================================================================
def record_health(name, ok, msg=""):
    health.append((name, ok, msg, "ok" if ok else "err"))

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
    ann_ps_model = load_model("ANN_PS_Model.keras")
    _jb = joblib
    ann_ps_proc = _ScalerShim(_jb.load("ANN_PS_Scaler_X.save"), _jb.load("ANN_PS_Scaler_y.save"))
    record_health("PS (ANN)", True, "loaded via .keras + joblib scalers")
except Exception as e:
    record_health("PS (ANN)", False, f"{e}")

ann_mlp_model = None; ann_mlp_proc = None
try:
    ann_mlp_model = load_model("ANN_MLP_Model.keras")
    _jb = joblib
    ann_mlp_proc = _ScalerShim(_jb.load("ANN_MLP_Scaler_X.save"), _jb.load("ANN_MLP_Scaler_y.save"))
    record_health("MLP (ANN)", True, "loaded via .keras + joblib scalers")
except Exception as e:
    record_health("MLP (ANN)", False, f"{e}")

try:
    rf_model = joblib.load("random_forest_model.pkl")
    record_health("Random Forest", True, "loaded")
except Exception as e:
    record_health("Random Forest", False, str(e))

try:
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model("XGBoost_trained_model_for_DI.json")
    record_health("XGBoost", True, "loaded")
except Exception as e:
    record_health("XGBoost", False, str(e))

try:
    cat_model = catboost.CatBoostRegressor(); cat_model.load_model("CatBoost.cbm")
    record_health("CatBoost", True, "loaded")
except Exception as e:
    cat_model = None; record_health("CatBoost", False, str(e))

def load_lightgbm_flex():
    cand = ["LightGBM_model", "LightGBM_model.txt", "LightGBM_model.bin",
            "LightGBM_model.pkl", "LightGBM_model.joblib"]
    for p in cand:
        if not Path(p).exists(): continue
        try: return lgb.Booster(model_file=p), "booster", p
        except Exception:
            try: return joblib.load(p), "sklearn", p
            except Exception: pass
    raise FileNotFoundError("No LightGBM_model file found.")

try:
    lgb_model, lgb_kind, lgb_path = load_lightgbm_flex()
    record_health("LightGBM", True, f"loaded as {lgb_kind} from {lgb_path}")
except Exception as e:
    lgb_model = None; record_health("LightGBM", False, str(e))

model_registry = {}
for name, ok, *_ in health:
    if not ok: continue
    if name == "XGBoost": model_registry["XGBoost"] = xgb_model
    elif name == "LightGBM" and lgb_model is not None: model_registry["LightGBM"] = lgb_model
    elif name == "CatBoost" and cat_model is not None: model_registry["CatBoost"] = cat_model
    elif name == "PS (ANN)" and ann_ps_model is not None: model_registry["PS"] = ann_ps_model
    elif name == "MLP (ANN)" and ann_mlp_model is not None: model_registry["MLP"] = ann_mlp_model
    elif name == "Random Forest": model_registry["Random Forest"] = rf_model

with st.sidebar:
    st.header("Model Health")
    for name, ok, msg, cls in health:
        st.markdown(f"- {'✅' if ok else '❌'} **{name}**<br/><small>{msg}</small>", unsafe_allow_html=True)

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# =============================================================================
# Ranges & Inputs
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

# Top-level responsive columns
left, right = st.columns([1.5, 1.8], gap="large")

with left:
    st.markdown("<div class='form-banner'>Inputs Features</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("<div class='section-header'>Geometry</div>", unsafe_allow_html=True)
        lw, hw, tw, b0, db, AR, M_Vlw = [num(*row) for row in GEOM]
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fc, fyt, fysh = [num(*row) for row in MATS[:3]]

    with c2:
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fyl, fybl = [num(*row) for row in MATS[3:]]
        st.markdown("<div class='section-header'>Reinf. Ratios</div>", unsafe_allow_html=True)
        rt, rsh, rl, rbl, s_db, axial, theta = [num(*row) for row in REINF]

with right:
    # small vertical nudge if needed
    st.markdown(f"<div style='height:{int(right_offset)}px'></div>", unsafe_allow_html=True)

    # hero image (no transform; keep within column)
    st.image("logo2-01.png", width=520, clamp=True)

    # model selector + actions
    row = st.columns([1.2, 2.2, 2.2], gap="small")

    with row[0]:
        available = list(model_registry.keys())
        order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
        ordered = [m for m in order if m in available] or ["(no models loaded)"]
        label_map = {"Random Forest": "RF"}
        display_labels = [label_map.get(m, m) for m in ordered]
        inv_label = {"RF": "Random Forest"}
        model_choice_label = st.selectbox("Model Selection", display_labels, key="model_select_compact")
        model_choice = inv_label.get(model_choice_label, model_choice_label)

    with row[1]:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        submit = st.button("Calculate", key="calc_btn")

    with row[2]:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        if st.button("Reset", key="reset_btn"):
            st.rerun()
        if st.button("Clear All", key="clear_btn"):
            st.session_state.results_df = pd.DataFrame()
            st.success("All predictions cleared.")

    # result + download
    badge_col, dl_col = st.columns([2.0, 1.4], gap="small")
    with badge_col: pred_banner = st.empty()
    with dl_col: dl_slot = st.empty()
    if not st.session_state.results_df.empty:
        csv = st.session_state.results_df.to_csv(index=False)
        dl_slot.download_button("📂 Download All Results as CSV", data=csv,
                                file_name="di_predictions.csv", mime="text/csv")

    chart_slot = st.empty()

# =============================================================================
# Prediction utilities & chart
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
        yhat = model_registry["PS"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_ps_proc.inverse_transform_y(yhat).item())

    elif choice == "MLP":
        Xn = ann_mlp_proc.transform_X(X)
        yhat = model_registry["MLP"].predict(Xn, verbose=0)[0][0]
        prediction = float(ann_mlp_proc.inverse_transform_y(yhat).item())

    else:
        raise RuntimeError("No trained model is available.")

    # Clamp to [0.035, 1.5]
    return max(0.035, min(float(prediction), 1.5))

def _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta_val):
    cols = ['l_w','h_w','t_w','f′c','fyt','fysh','fyl','fybl','ρt','ρsh','ρl','ρbl','P/(Agf′c)','b0','db','s/db','AR','M/Vlw','θ']
    x = np.array([[lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta_val]], dtype=np.float32)
    return pd.DataFrame(x, columns=cols)

def _sweep_curve_df(model_choice, base_df, theta_max=THETA_MAX, step=0.1):
    if model_choice not in model_registry: return pd.DataFrame(columns=["θ","Predicted_DI"])
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
                    theta_max: float = THETA_MAX, di_max: float = 1.5, size: int = 520):
    import altair as alt
    AXIS_LABEL_FS = 14
    AXIS_TITLE_FS = 16

    x_ticks = np.linspace(0.0, theta_max, 5).round(2)
    base_axes_df = pd.DataFrame({"θ": [0.0, theta_max], "Predicted_DI": [0.0, 0.0]})
    axes_layer = (
        alt.Chart(base_axes_df).mark_line(opacity=0).encode(
            x=alt.X("θ:Q", title="Drift Ratio (θ)",
                    scale=alt.Scale(domain=[0, theta_max], nice=False, clamp=True),
                    axis=alt.Axis(values=list(x_ticks), labelFontSize=AXIS_LABEL_FS, titleFontSize=AXIS_TITLE_FS)),
            y=alt.Y("Predicted_DI:Q", title="Damage Index (DI)",
                    scale=alt.Scale(domain=[0, di_max], nice=False, clamp=True),
                    axis=alt.Axis(values=[0.0, 0.2, 0.5, 1.0, 1.5], labelFontSize=AXIS_LABEL_FS, titleFontSize=AXIS_TITLE_FS)),
        ).properties(width=size, height=size)
    )
    line_layer = alt.Chart(curve_df).mark_line(strokeWidth=3).encode(x="θ:Q", y="Predicted_DI:Q").properties(width=size, height=size)
    k = 3
    pts = curve_df.iloc[::k].copy() if not curve_df.empty else pd.DataFrame({"θ": [], "Predicted_DI": []})
    points_layer = alt.Chart(pts).mark_circle(size=70, opacity=0.8).encode(
        x="θ:Q", y="Predicted_DI:Q",
        tooltip=[alt.Tooltip("θ:Q", title="θ", format=".2f"), alt.Tooltip("Predicted_DI:Q", title="DI", format=".4f")]
    )
    st.altair_chart(alt.layer(axes_layer, line_layer, points_layer).configure_view(strokeWidth=0), use_container_width=False)

# =============================================================================
# Predict on click; always render curve
# =============================================================================
_order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
def _pick_default_model():
    for m in _order:
        if m in model_registry: return m
    return None

label_in_state = st.session_state.get("model_select_compact")
model_choice = {"RF": "Random Forest"}.get(label_in_state, label_in_state) or _pick_default_model()

if (model_choice is None) or (model_choice not in model_registry):
    st.error("No trained model is available. Please check the Model Selection on the right.")
else:
    xdf = _make_input_df(
        lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta
    )
    if submit:
        try:
            pred = predict_di(model_choice, None, xdf)
            row = xdf.copy(); row["Predicted_DI"] = pred
            st.session_state.results_df = pd.concat([st.session_state.results_df, row], ignore_index=True)
            pred_banner.markdown(f"<div class='prediction-result'>Predicted Damage Index (DI): {pred:.4f}</div>", unsafe_allow_html=True)
            csv = st.session_state.results_df.to_csv(index=False)
            dl_slot.download_button("📂 Download All Results as CSV", data=csv, file_name="di_predictions.csv", mime="text/csv", key="dl_csv_after_submit")
        except Exception as e:
            st.error(f"Prediction failed for {model_choice}: {e}")

    _curve_df = _sweep_curve_df(model_choice, xdf, theta_max=THETA_MAX, step=0.1)
    with chart_slot:
        render_di_chart(st.session_state.results_df, _curve_df, theta_max=THETA_MAX, di_max=1.5, size=520)

# =============================================================================
# Recent predictions (optional)
# =============================================================================
with st.sidebar:
    show_recent = st.checkbox("Show Recent Predictions", value=False)
if show_recent and not st.session_state.results_df.empty:
    st.markdown("### 🧾 Recent Predictions")
    for i, row in st.session_state.results_df.tail(5).reset_index(drop=True).iterrows():
        st.markdown(f"<span class='recent-box'>Pred {i+1} ➔ DI = {row['Predicted_DI']:.4f}</span>", unsafe_allow_html=True)
