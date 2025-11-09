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

# =============================================================================
# 🎨 STEP 3: STREAMLIT PAGE CONFIGURATION & UI STYLING
# =============================================================================
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

# ====== ONLY FONTS/LOGO KNOBS BELOW (smaller defaults) ======
SCALE_UI = 0.36  # global shrink (pure scaling; lower => smaller). Safe at 100% zoom.

s = lambda v: int(round(v * SCALE_UI))

FS_TITLE   = s(100)  # page title
FS_SECTION = s(60)  # section headers
FS_LABEL   = s(50)  # input & select labels (katex included)
FS_UNITS   = s(30)  # math units in labels
FS_INPUT   = s(30)  # number input value
FS_SELECT  = s(35)  # dropdown value/options
FS_BUTTON  = s(20)  # Calculate / Reset / Clear All
FS_BADGE   = s(30)  # predicted badge
FS_RECENT  = s(20)  # small chips
INPUT_H    = max(32, int(FS_INPUT * 2.0))

# header logo default height (can still be changed by URL param "logo")
DEFAULT_LOGO_H = 60

PRIMARY   = "#8E44AD"
SECONDARY = "#f9f9f9"
INPUT_BG     = "#ffffff"
INPUT_BORDER = "#e6e9f2"
LEFT_BG      = "#e0e4ec"

# =============================================================================
# 🎨 STEP 3.1: COMPREHENSIVE CSS STYLING & THEME SETUP
# =============================================================================
# Adjusting CSS to make buttons and model selection box appear in the same line and fixed issues
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
  div[data-testid="stNumberInput"] [data-baseweb*="input"]:hover {{ border-color:#d6dced !important; }}
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
  div[data-testid="stNumberInput"] button:hover {{ border-color:#cbd3e5 !important; }}

  /* Adjusting Model Selection */
  .stSelectbox [role="combobox"],
  div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child,
  div[data-testid="stSelectbox"] div[role="listbox"],
  div[data-testid="stSelectbox"] div[role="option"] {{
      font-size:{FS_SELECT}px !important;
      width: 100% !important; /* Fixing the width of the model selection box */
  }}

  /* Button Fixes */
  div.stButton > button {{
    font-size:{FS_BUTTON}px !important;
    height:{max(42, int(round(FS_BUTTON*1.45)))}px !important;
    line-height:{max(36, int(round(FS_BUTTON*1.15)))}px !important;
    white-space:nowrap !important;
    color:#fff !important;
    font-weight:700; border:none !important; border-radius:8px !important;
    background:#4CAF50 !important;
  }}
  div.stButton > button:hover {{ filter: brightness(0.95); }}

  /* Adjusting Button Placement in the same line */
  #action-row {{
    display: flex;
    gap: 10px;
    justify-content: flex-start;
    align-items: center;
  }}

  #action-row .stButton {{
    margin-bottom: 10px;
  }}

  /* Adjusting download button visibility: only show after calculation */
  .download-csv {
    display: none !important;
  }

  .download-csv.show {
    display: inline-block !important;
  }

  .prediction-result {{
    font-size:{FS_BADGE}px !important; font-weight:700; color:#2e86ab;
    background:#f1f3f4; padding:.6rem; border-radius:6px; text-align:center; margin-top:.6rem;
  }}
</style>
""")

# =============================================================================
# 🎯 STEP 4: INTERFACE POSITIONING & LAYOUT ADJUSTMENTS
# =============================================================================

# Adjust the layout positioning for buttons and fields if needed
# This code ensures everything remains in line, and no elements are hidden or misplaced.

# =============================================================================
# ⚙️ STEP 5: FEATURE FLAGS & SIDEBAR TUNING CONTROLS
# =============================================================================
def _is_on(v): return str(v).lower() in {"1","true","yes","on"}
SHOW_TUNING = _is_on(os.getenv("SHOW_TUNING", "0"))
try:
    qp = st.query_params
    if "tune" in qp:
        SHOW_TUNING = _is_on(qp.get("tune"))
except Exception:
    try:
        qp = st.experimental_get_query_params()
        if "tune" in qp:
            SHOW_TUNING = _is_on(qp.get("tune", ["0"])[0])
    except Exception:
        pass

# =============================================================================
# 🏷️ STEP 6: DYNAMIC HEADER & LOGO POSITIONING
# =============================================================================
try:
    _logo_path = BASE_DIR / "TJU logo.png"
    _b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii") if _logo_path.exists() else ""
except Exception:
    _b64 = ""

# Adding the fix for button and model selection display
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='form-banner'>Inputs Features</div>", unsafe_allow_html=True)
    st.markdown("<style>.section-header{margin:.2rem 0 !important;}</style>", unsafe_allow_html=True)
    css("<div id='leftwrap'>")
    css("<div id='compact-form'>")

    # ⬇️ Three columns: Geometry | Reinf. Ratios | Material Strengths
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")

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

    css("</div>")
    css("</div>")


# =============================================================================
# 🎮 STEP 9: RIGHT PANEL - CONTROLS & INTERACTION ELEMENTS
# =============================================================================
HERO_X, HERO_Y, HERO_W = 100, 5, 300
MODEL_X, MODEL_Y = 100, -2
CHART_W = 300

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

    # Show Download button after Calculate click
    if submit:
        if not st.session_state.results_df.empty:
            csv = st.session_state.results_df.to_csv(index=False)
            dl_slot.download_button("📂 Download as CSV", data=csv, file_name="di_predictions.csv", mime="text/csv", use_container_width=False, key="dl_csv_main")

    col1, col2 = st.columns([0.01, 20])
    with col2:
        chart_slot = st.empty()

# Finalize layout fixes


# =============================================================================
# 🔮 STEP 10: PREDICTION ENGINE & CURVE GENERATION UTILITIES
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
                    theta_max: float = THETA_MAX, di_max: float = 1.5, size: int = 460):
    import altair as alt
    selection = alt.selection_point(name='select', fields=['θ', 'Predicted_DI'], nearest=True, on='mouseover', empty=False, clear='mouseout')
    AXIS_LABEL_FS = 14; AXIS_TITLE_FS = 16; TICK_SIZE = 6; TITLE_PAD = 10; LABEL_PAD = 6
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

    points_layer = alt.Chart(curve_points).mark_circle(size=60, opacity=0.7).encode(
        x="θ:Q", y="Predicted_DI:Q",
        tooltip=[alt.Tooltip("θ:Q", title="Drift Ratio (θ)", format=".2f"),
                 alt.Tooltip("Predicted_DI:Q", title="Predicted DI", format=".4f")]
    ).add_params(selection)

    rules_layer = alt.Chart(curve).mark_rule(color='red', strokeWidth=2).encode(x="θ:Q", y="Predicted_DI:Q").transform_filter(selection)
    text_layer = alt.Chart(curve).mark_text(align='left', dx=8, dy=-8, fontSize=14, fontWeight='bold', color='red').encode(
        x="θ:Q", y="Predicted_DI:Q", text=alt.Text("Predicted_DI:Q", format=".4f")
    ).transform_filter(selection)

    chart = (alt.layer(axes_layer, line_layer, points_layer, rules_layer, text_layer)
             .configure_view(strokeWidth=0)
             .configure_axis(domain=True, ticks=True)
             .configure(padding={"left": 6, "right": 6, "top": 6, "bottom": 6}))
    chart_html = chart.to_html()
    chart_html = chart_html.replace('</style>',
        '</style><style>.vega-embed .vega-tooltip, .vega-embed .vega-tooltip * { font-size: 14px !important; font-weight: bold !important; background: #000 !important; color: #fff !important; padding: 12px !important; }</style>')
    st.components.v1.html(chart_html, height=size + 100)

# =============================================================================
# ⚡ STEP 11: PREDICTION EXECUTION & REAL-TIME VISUALIZATION
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
# 🎨 STEP 12: FINAL UI POLISH & BANNER STYLING
# =============================================================================
st.markdown("""
<style>
.form-banner{
  background: linear-gradient(90deg, #0E9F6E, #84CC16) !important;
  color: #fff !important;
  text-align: center !important;
  border-radius: 10px !important;
  padding: .45rem .75rem !important;
  margin-top: 65px !important;
  transform: translateY(0) !important;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 📋 STEP 13: RECENT PREDICTIONS DISPLAY (OPTIONAL)
# =============================================================================
if SHOW_TUNING and _show_recent and not st.session_state.results_df.empty:
    right_predictions = st.empty()
    with right_predictions:
        st.markdown("### 🧾 Recent Predictions")
        for i, row in st.session_state.results_df.tail(5).reset_index(drop=True).iterrows():
            st.markdown(
                f"<div class='recent-box' style='display:inline-block; width:auto; padding:4px 10px;'>"
                f"Pred {i+1} ➔ DI = {row['Predicted_DI']:.4f}</div>",
                unsafe_allow_html=True
            )

# =============================================================================
# 🎛️ STEP 14: DYNAMIC STYLE OVERRIDES VIA QUERY PARAMETERS
# =============================================================================
def _get_qp():
    try:
        return st.query_params
    except Exception:
        try:
            return st.experimental_get_query_params()
        except Exception:
            return {}

_qp = _get_qp()

def _get_int(name):
    try:
        v = _qp.get(name)
        if isinstance(v, list): v = v[0]
        return int(v) if v not in (None, "", []) else None
    except Exception:
        return None

_FS_TITLE   = _get_int("fs_title")
_FS_SECTION = _get_int("fs_section")
_FS_LABEL   = _get_int("fs_label")
_FS_UNITS   = _get_int("fs_units")
_FS_INPUT   = _get_int("fs_input")
_FS_SELECT  = _get_int("fs_select")
_FS_BUTTON  = _get_int("fs_button")
_FS_BADGE   = _get_int("fs_badge")
_FS_RECENT  = _get_int("fs_recent")
_LOGO_H     = _get_int("logo")

_rules = []
if _FS_TITLE   is not None: _rules.append(f".page-header__title{{font-size:{_FS_TITLE}px !important;}}")
if _FS_SECTION is not None: _rules.append(f".section-header{{font-size:{_FS_SECTION}px !important;}}")
if _FS_LABEL   is not None: _rules.append(f".stNumberInput label, .stSelectbox label{{font-size:{_FS_LABEL}px !important;}}")
if _FS_UNITS   is not None: _rules.append(f".stNumberInput label .katex .mathrm, .stSelectbox label .katex .mathrm{{font-size:{_FS_UNITS}px !important;}}")
if _FS_INPUT   is not None: _rules.append(f"div[data-testid='stNumberInput'] input{{font-size:{_FS_INPUT}px !important;}}")
if _FS_SELECT  is not None:
    _rules.append(f".stSelectbox [role='combobox'], div[data-testid='stSelectbox'] div[data-baseweb='select'] > div > div:first-child{{font-size:{_FS_SELECT}px !important;}}")
    _rules.append(f"div[data-testid='stSelectbox'] div[role='listbox'], div[data-testid='stSelectbox'] div[role='option']{{font-size:{_FS_SELECT}px !important;}}")
if _FS_BUTTON  is not None:
    _btn_h  = max(42, int(round(_FS_BUTTON * 1.45)))
    _btn_lh = max(36, int(round(_FS_BUTTON * 1.15)))
    _rules.append(f"div.stButton > button{{font-size:{_FS_BUTTON}px !important;height:{_btn_h}px !important;line-height:{_btn_lh}px !important;white-space:nowrap !important;}}")
else:
    _rules.append("div.stButton > button{white-space:nowrap !important;}")

if _FS_BADGE  is not None: _rules.append(f".prediction-result{{font-size:{_FS_BADGE}px !important;}}")
if _FS_RECENT is not None: _rules.append(f".recent-box{{font-size:{_FS_RECENT}px !important;}}")
if _LOGO_H    is not None: _rules.append(f".page-header__logo{{height:{_LOGO_H}px !important;}}")

if _rules:
    css("<style id='late-font-logo-overrides'>" + "\n".join(_rules) + "</style>")
# =============================================================================
# ✅ COMPLETED: RC SHEAR WALL DI ESTIMATOR APPLICATION
# =============================================================================

