# --- Streamlit launcher for Spyder: must be at the VERY TOP, before any `import streamlit` ---
if __name__ == "__main__":
    import os, sys, subprocess
    from pathlib import Path
    if os.environ.get("__ST_LAUNCHED_FROM_SPYDER__", "") != "1":
        env = os.environ.copy()
        env["__ST_LAUNCHED_FROM_SPYDER__"] = "1"
        this_file = str(Path(__file__).resolve())
        cmd = [sys.executable, "-m", "streamlit", "run", this_file, "--server.headless", "false"]
        subprocess.Popen(cmd, env=env)
        raise SystemExit(0)
# --- end launcher ---

# -*- coding: utf-8 -*-

# =============================================================================
# Imports (unchanged)
# =============================================================================
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb

from tensorflow.keras.models import load_model

css = lambda s: st.markdown(s, unsafe_allow_html=True)
def b64(path: Path) -> str: return base64.b64encode(path.read_bytes()).decode("ascii")

# =============================================================================
# Page config (unchanged)
# =============================================================================
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

# =============================================================================
# GLOBAL STYLES — responsive & stable across screens
#   (Replace fixed px with clamp() and remove negative/absolute offsets)
# =============================================================================
css("""
<style>
  :root{
    /* Responsive font scales */
    --fs-title: clamp(26px, 3.2vw, 42px);
    --fs-section: clamp(18px, 2.1vw, 28px);
    --fs-label: clamp(14px, 1.6vw, 22px);
    --fs-units: clamp(12px, 1.2vw, 16px);
    --fs-input: clamp(13px, 1.4vw, 18px);
    --fs-select: clamp(14px, 1.6vw, 20px);
    --fs-button: clamp(14px, 1.6vw, 20px);
    --fs-badge: clamp(13px, 1.4vw, 18px);
    --fs-recent: clamp(11px, 1.2vw, 16px);
  }

  .block-container { padding-top: 0rem; max-width: 1400px; }

  /* Title row */
  .page-header {
    display:flex; align-items:center; gap:16px; margin:0 0 .25rem 0;
  }
  .page-header__title {
    font-size: var(--fs-title); font-weight:800; margin:0;
  }
  .page-header__logo { height: clamp(36px, 5vw, 70px); width:auto; display:block; }

  /* Section headers */
  .section-header{
    font-size: var(--fs-section) !important; font-weight: 700; margin:.35rem 0;
  }

  /* Number inputs / labels */
  .stNumberInput label, .stSelectbox label { font-size: var(--fs-label) !important; font-weight:700; }
  .stNumberInput label .katex, .stSelectbox label .katex { font-size: var(--fs-label) !important; line-height:1.1 !important; }
  .stNumberInput label .katex .mathrm, .stSelectbox label .katex .mathrm { font-size: var(--fs-units) !important; }

  div[data-testid="stNumberInput"] input[type="number"],
  div[data-testid="stNumberInput"] input[type="text"]{
    font-size: var(--fs-input) !important; height: clamp(36px, 4.2vw, 48px) !important;
    font-weight:600 !important; padding:8px 10px !important;
  }
  div[data-testid="stNumberInput"] [data-baseweb*="input"]{
    background:#fff !important; border:1px solid #e6e9f2 !important; border-radius:12px !important;
    box-shadow:0 1px 2px rgba(16,24,40,.06) !important;
  }

  /* Selectbox value/options */
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child { font-size: var(--fs-select) !important; }
  div[data-testid="stSelectbox"] div[role="listbox"] div[role="option"] { font-size: var(--fs-select) !important; }

  /* Buttons */
  div.stButton > button{
    font-size: var(--fs-button) !important; height: clamp(36px, 4.2vw, 46px) !important;
    color:#fff !important; font-weight:700; border:none !important; border-radius:8px !important;
    background:#4CAF50 !important;
  }
  button[key="reset_btn"]{ background:#2196F3 !important; }
  button[key="clear_btn"]{ background:#f44336 !important; }

  /* Banner */
  .form-banner{
    text-align:center; background: linear-gradient(90deg, #0E9F6E, #84CC16); color:#fff;
    padding:.45rem .75rem; border-radius:10px; font-weight:800;
    font-size: calc(var(--fs-section) + 2px); margin:.35rem 0 .6rem 0 !important;
  }

  .prediction-result{
    font-size: var(--fs-badge) !important; font-weight:700; color:#2e86ab;
    background:#f1f3f4; padding:.5rem .65rem; border-radius:6px; text-align:center; margin-top:.35rem;
    white-space:nowrap; display:inline-block;
  }

  /* Remove Streamlit top header spacing */
  header[data-testid="stHeader"]{ height:0 !important; padding:0 !important; background:transparent !important; }
  header[data-testid="stHeader"] *{ display:none !important; }

  /* Make Altair fill container width */
  .altair-chart-wrap{ width:100%; }
</style>
""")

# =============================================================================
# Title + Logos (kept, responsive)
# =============================================================================
try:
    _logo_path = Path(__file__).resolve().parent / "TJU logo.png"
    _b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii") if _logo_path.exists() else ""
except Exception:
    _b64 = ""

st.markdown(
    f"""
    <div class="page-header">
      <div class="page-header__title">Predict Damage index (DI) for RC Shear Walls</div>
      {f'<img class="page-header__logo" alt="Logo" src="data:image/png;base64,{_b64}" />' if _b64 else ''}
    </div>
    """,
    unsafe_allow_html=True
)

# Optional: small header tweaks collapsed by default (so the sidebar isn’t intrusive)
with st.sidebar.expander("Header options (optional)", expanded=False):
    title_x = st.number_input("Title X offset (px)", -200, 200, 0, 2)
    title_y = st.number_input("Title Y offset (px)", -120, 120, 0, 2)
    logo_h  = st.number_input("Logo height (px)", 20, 200, 80, 2)
    css(f"""
    <style>
      .page-header__title{{ transform: translate({int(title_x)}px,{int(title_y)}px); }}
      .page-header__logo{{ height:{int(logo_h)}px; }}
    </style>
    """)

# =============================================================================
# Model loading (unchanged)
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
    ann_ps_model = load_model("ANN_PS_Model.keras")
    import joblib as _jb
    ann_ps_proc = _ScalerShim(_jb.load("ANN_PS_Scaler_X.save"), _jb.load("ANN_PS_Scaler_y.save"))
    record_health("PS (ANN)", True, "loaded via .keras + joblib scalers")
except Exception as e:
    record_health("PS (ANN)", False, f"{e}")

ann_mlp_model = None; ann_mlp_proc = None
try:
    ann_mlp_model = load_model("ANN_MLP_Model.keras")
    import joblib as _jb
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

with st.sidebar.expander("Model Health", expanded=False):
    for name, ok, msg, cls in health:
        st.markdown(f"- {'✅' if ok else '❌'} **{name}**  \n<small>{msg}</small>", unsafe_allow_html=True)

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# =============================================================================
# Ranges & Inputs (unchanged values; only removed negative offsets)
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

def dv(R, key, proposed): lo, hi = R[key]; return float(max(lo, min(proposed, hi)))

def num(label, key, default, step, fmt, help_):
    return st.number_input(
        label, value=dv(R, key, default), step=step,
        min_value=R[key][0], max_value=R[key][1],
        format=fmt if fmt else None, help=help_
    )

# Layout columns: left form (2 cols) + right visual
left, right = st.columns([1.4, 1.6], gap="large")

with left:
    st.markdown("<div class='form-banner'>Inputs Features</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("<div class='section-header'>Geometry </div>", unsafe_allow_html=True)
        lw, hw, tw, b0, db, AR, M_Vlw = [num(*row) for row in GEOM]
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fc, fyt, fysh = [num(*row) for row in MATS[:3]]

    with c2:
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fyl, fybl = [num(*row) for row in MATS[3:]]
        st.markdown("<div class='section-header'>Reinf. Ratios </div>", unsafe_allow_html=True)
        rt, rsh, rl, rbl, s_db, axial, theta = [num(*row) for row in REINF]

with right:
    # wall sketch
    try:
        img_b64 = b64(Path("logo2-01.png"))
        st.markdown(f"<img src='data:image/png;base64,{img_b64}' style='height:auto; width: clamp(260px, 32vw, 420px);'/>",
                    unsafe_allow_html=True)
    except Exception:
        pass

    # model selector + actions
    row = st.columns([1.2, 1, 1, 1.1], gap="small")
    with row[0]:
        available = set(model_registry.keys())
        order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
        ordered_keys = [m for m in order if m in available] or ["(no models loaded)"]
        display_labels = ["RF" if m == "Random Forest" else m for m in ordered_keys]
        _label_to_key = {"RF": "Random Forest"}
        model_choice_label = st.selectbox("Model Selection", display_labels, key="model_select_compact")
        model_choice = _label_to_key.get(model_choice_label, model_choice_label)

    with row[1]:
        submit = st.button("Calculate", key="calc_btn", use_container_width=True)
    with row[2]:
        if st.button("Reset", key="reset_btn", use_container_width=True):
            st.rerun()
    with row[3]:
        if st.button("Clear All", key="clear_btn", use_container_width=True):
            st.session_state.results_df = pd.DataFrame()
            st.success("All predictions cleared.")

    badge_col, dl_col = st.columns([2, 1], gap="small")
    with badge_col:
        pred_banner = st.empty()
    with dl_col:
        dl_slot = st.empty()

    chart_slot = st.container()

# =============================================================================
# Prediction + Curve (unchanged math; chart width responsive)
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

    return max(0.035, min(prediction, 1.5))

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

def render_di_chart(results_df: pd.DataFrame, curve_df: pd.DataFrame, theta_max: float = THETA_MAX, di_max: float = 1.5):
    import altair as alt
    base_axes_df = pd.DataFrame({"θ": [0.0, theta_max], "Predicted_DI": [0.0, 0.0]})
    x_ticks = np.linspace(0.0, theta_max, 5).round(2)

    axes_layer = (
        alt.Chart(base_axes_df).mark_line(opacity=0).encode(
            x=alt.X("θ:Q", title="Drift Ratio (θ)", scale=alt.Scale(domain=[0, theta_max], nice=False, clamp=True),
                    axis=alt.Axis(values=list(x_ticks))),
            y=alt.Y("Predicted_DI:Q", title="Damage Index (DI)", scale=alt.Scale(domain=[0, di_max], nice=False, clamp=True),
                    axis=alt.Axis(values=[0.0, 0.2, 0.5, 1.0, 1.5])),
        ).properties(width="container", height=420)
    )

    curve = curve_df if (curve_df is not None and not curve_df.empty) else pd.DataFrame({"θ": [], "Predicted_DI": []})
    line_layer = alt.Chart(curve).mark_line(point=True).encode(x="θ:Q", y="Predicted_DI:Q").properties(width="container", height=420)

    chart = (alt.layer(axes_layer, line_layer)
             .configure_view(strokeWidth=0)
             .interactive())

    st.markdown('<div class="altair-chart-wrap">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Run prediction / render
_order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
_label_to_key = {"RF": "Random Forest"}

label_from_state = (st.session_state.get("model_select_compact") or st.session_state.get("model_select"))
model_choice = _label_to_key.get(label_from_state, label_from_state) if label_from_state else next((m for m in _order if m in model_registry), None)

if (model_choice is None) or (model_choice not in model_registry):
    st.error("No trained model is available. Please check the Model Selection on the right.")
else:
    if submit:
        xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
        try:
            pred = predict_di(model_choice, None, xdf)
            row = xdf.copy(); row["Predicted_DI"] = pred
            st.session_state.results_df = pd.concat([st.session_state.results_df, row], ignore_index=True)
            pred_banner.markdown(f"<div class='prediction-result'>Predicted Damage Index (DI): {pred:.4f}</div>", unsafe_allow_html=True)
            csv = st.session_state.results_df.to_csv(index=False)
            dl_slot.download_button("📂 Download All Results as CSV", data=csv, file_name="di_predictions.csv",
                                    mime="text/csv", use_container_width=True, key="dl_csv_after_submit")
        except Exception as e:
            st.error(f"Prediction failed for {model_choice}: {e}")

    _base_xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
    _curve_df = _sweep_curve_df(model_choice, _base_xdf, theta_max=THETA_MAX, step=0.1)

with right:
    render_di_chart(st.session_state.results_df, _curve_df, theta_max=THETA_MAX, di_max=1.5)

# Optional recent predictions (collapsed)
with st.sidebar.expander("Recent Predictions", expanded=False):
    if not st.session_state.results_df.empty:
        for i, row in st.session_state.results_df.tail(5).reset_index(drop=True).iterrows():
            st.write(f"Pred {i+1} → DI = {row['Predicted_DI']:.4f}")
