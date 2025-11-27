DOC_NOTES = """
RC Shear Wall Damage Index (DI) Estimator — compact, same logic/UI
"""

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

# Session state initialization
st.session_state.setdefault("results_df", pd.DataFrame())

# Utility functions
css = lambda s: st.markdown(s, unsafe_allow_html=True)
def b64(path: Path) -> str: return base64.b64encode(path.read_bytes()).decode("ascii")
def dv(R, key, proposed): lo, hi = R[key]; return float(max(lo, min(proposed, hi)))

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

# Page configuration
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

# Main CSS Styling
css("""
<style>
html, body{ margin:0 !important; padding:0 !important; }
header[data-testid="stHeader"]{ height:0 !important; padding:0 !important; background:transparent !important; }
header[data-testid="stHeader"] *{ display:none !important; }
div.stApp{ margin-top:-2rem !important; }
section.main > div.block-container{ padding-top:0.5rem !important; margin-top:0 !important; }

.block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
h1 { font-size:20px !important; margin:0 rem 0 !important; }

.section-header { font-size:60px !important; font-weight:700; margin:.35rem 0; }
.stNumberInput label, .stSelectbox label { font-size:50px !important; font-weight:700; }
.stNumberInput label .katex, .stSelectbox label .katex { font-size:50px !important; line-height:1.2 !important; }
.stNumberInput label .katex .mathrm, .stSelectbox label .katex .mathrm { font-size:30px !important; }

div[data-testid="stNumberInput"] input[type="number"], div[data-testid="stNumberInput"] input[type="text"] {
    font-size:30px !important; height:60px !important; line-height:52px !important; font-weight:600 !important;
    padding:10px 12px !important;
}

div[data-testid="stNumberInput"] [data-baseweb*="input"] {
    background:#ffffff !important; border:1px solid #e6e9f2 !important; border-radius:12px !important;
    box-shadow:0 1px 2px rgba(16,24,40,.06) !important; transition:border-color .15s ease, box-shadow .15s ease !important;
}
div[data-testid="stNumberInput"] [data-baseweb*="input"]:hover { border-color:#d6dced !important; }
div[data-testid="stNumberInput"] [data-baseweb*="input"]:focus-within {
    border-color:#8E44AD !important; box-shadow:0 0 0 3px rgba(106,17,203,.15) !important;
}

div[data-testid="stNumberInput"] button { display: none !important; }

.stSelectbox [role="combobox"], div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:first-child,
div[data-testid="stSelectbox"] div[role="listbox"], div[data-testid="stSelectbox"] div[role="option"] {
    font-size:35px !important;
}

div.stButton > button {
    font-size:20px !important; height:42px !important; line-height:36px !important; white-space:nowrap !important;
    color:#fff !important; font-weight:700; border:none !important; border-radius:8px !important;
}
button[key="calc_btn"] { background:#4CAF50 !important; }
button[key="reset_btn"] { background:#2196F3 !important; }
button[key="clear_btn"] { background:#f44336 !important; }

.form-banner {
    text-align:center; background: linear-gradient(90deg, #0E9F6E, #84CC16); color: #fff;
    padding:.45rem .75rem; border-radius:10px; font-weight:800; font-size:64px; margin:.1rem 0 !important;
}

.prediction-with-color {
    color: #2e86ab !important; font-weight: 700 !important; font-size: 30px !important;
    background: #f1f3f4 !important; padding: 10px 12px !important; border-radius: 6px !important;
    text-align: center !important; margin: 0 !important; height: 45px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    width: 180px !important;
}

/* Full page left side gray background */
html, body, #root, .stApp, section.main, .block-container, [data-testid="stAppViewContainer"] {
    background: linear-gradient(90deg, #e0e4ec 60%, transparent 60%) !important;
    min-height: 100vh !important; height: auto !important;
}

/* Selectbox styling - grey theme */
div[data-testid="stSelectbox"] [data-baseweb="select"] {
    border: none !important; box-shadow: none !important; background: #D3D3D3 !important;
    height: 40px !important; width: 180px !important; border-radius: 8px !important;
    padding: 0px 12px !important; outline: none !important;
}
div[data-testid="stSelectbox"] > div { border: none !important; box-shadow: none !important; outline: none !important; }
div[data-testid="stSelectbox"] > div > div { height: 40px !important; display: flex !important; align-items: center !important; }
div[data-testid="stSelectbox"] label p { font-size: 50px !important; color: black !important; font-weight: bold !important; }

[data-baseweb="select"] *, [data-baseweb="popover"] *, [data-baseweb="menu"] *,
[data-baseweb="select"] [role="listbox"], [data-baseweb="select"] [role="combobox"] { 
    color: black !important; background-color: #D3D3D3 !important; font-size: 35px !important; 
    border: none !important; outline: none !important; box-shadow: none !important;
}

div[role="option"] { color: black !important; font-size: 35px !important; background-color: #D3D3D3 !important; }
div[role="option"]:hover { background-color: #B8B8B8 !important; color:black !important; }

/* Logo positioning */
.page-header-outer { position: fixed !important; top: 0 !important; right: 0 !important; width: 100% !important; height: 0 !important; z-index: 9999 !important; }
.page-header { display: flex !important; justify-content: flex-end !important; align-items: flex-start !important; width: 100% !important; height: 0 !important; }
.page-header__logo { height: 45px !important; width: auto !important; position: fixed !important; top: 35px !important; right: 200px !important; z-index: 9999 !important; }
.main .block-container { padding-top: 100px !important; }

/* CHART POSITIONING - MOVED UP */
[data-testid="column"]:last-child {
    margin-top: -450px !important;
    padding-top: 0px !important;
}
[data-testid="column"]:last-child .element-container {
    margin-top: -450px !important;
    padding-top: 0px !important;
}
</style>
""")

# Logo setup
try:
    _logo_path = BASE_DIR / "TJU logo.png"
    _b64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii") if _logo_path.exists() else ""
except Exception:
    _b64 = ""

if _b64:
    css(f"""
    <div class="page-header-outer">
      <div class="page-header">
        <img class="page-header__logo" alt="Logo" src="data:image/png;base64,{_b64}" />
      </div>
    </div>
    """)

# Model loading
class _ScalerShim:
    def __init__(self, X_scaler, y_scaler):
        import numpy as _np
        self._np = _np
        self.Xs = X_scaler
        self.Ys = y_scaler
    def transform_X(self, X): return self.Xs.transform(X)
    def inverse_transform_y(self, y):
        y = self._np.array(y).reshape(-1, 1)
        return self.Ys.inverse_transform(y)

model_registry = {}

# Load all models
try:
    ps_model_path = pfind(["ANN_PS_Model.keras", "ANN_PS_Model.h5"])
    ann_ps_model = _load_keras_model(ps_model_path)
    sx = joblib.load(pfind(["ANN_PS_Scaler_X.save","ANN_PS_Scaler_X.pkl","ANN_PS_Scaler_X.joblib"]))
    sy = joblib.load(pfind(["ANN_PS_Scaler_y.save","ANN_PS_Scaler_y.pkl","ANN_PS_Scaler_y.joblib"]))
    ann_ps_proc = _ScalerShim(sx, sy)
    model_registry["PS"] = ann_ps_model
except Exception: pass

try:
    mlp_model_path = pfind(["ANN_MLP_Model.keras", "ANN_MLP_Model.h5"])
    ann_mlp_model = _load_keras_model(mlp_model_path)
    sx = joblib.load(pfind(["ANN_MLP_Scaler_X.save","ANN_MLP_Scaler_X.pkl","ANN_MLP_Scaler_X.joblib"]))
    sy = joblib.load(pfind(["ANN_MLP_Scaler_y.save","ANN_MLP_Scaler_y.pkl","ANN_MLP_Scaler_y.joblib"]))
    ann_mlp_proc = _ScalerShim(sx, sy)
    model_registry["MLP"] = ann_mlp_model
except Exception: pass

try:
    rf_path = pfind(["random_forest_model.pkl", "random_forest_model.joblib", "rf_model.pkl", "RF_model.pkl"])
    rf_model = joblib.load(rf_path)
    model_registry["Random Forest"] = rf_model
except Exception: pass

try:
    xgb_path = pfind(["XGBoost_trained_model_for_DI.json","Best_XGBoost_Model.json","xgboost_model.json"])
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model(xgb_path)
    model_registry["XGBoost"] = xgb_model
except Exception: pass

try:
    cat_path = pfind(["CatBoost.cbm","Best_CatBoost_Model.cbm","catboost.cbm"])
    cat_model = catboost.CatBoostRegressor(); cat_model.load_model(cat_path)
    model_registry["CatBoost"] = cat_model
except Exception: pass

try:
    p = pfind(["LightGBM_model.txt","Best_LightGBM_Model.txt","LightGBM_model.bin"])
    lgb_model = lgb.Booster(model_file=str(p))
    model_registry["LightGBM"] = lgb_model
except Exception: pass

# Input parameters
R = {
    "lw":(400.0,3500.0), "hw":(495.0,5486.4), "tw":(26.0,305.0), "fc":(13.38,93.6),
    "fyt":(0.0,1187.0), "fysh":(0.0,1375.0), "fyl":(160.0,1000.0), "fybl":(0.0,900.0),
    "rt":(0.000545,0.025139), "rsh":(0.0,0.041888), "rl":(0.0,0.029089), "rbl":(0.0,0.031438),
    "axial":(0.0,0.86), "b0":(45.0,3045.0), "db":(0.0,500.0), "s_db":(0.0,47.65625),
    "AR":(0.388889,5.833333), "M_Vlw":(0.388889,4.1), "theta":(0.0275,4.85),
}
THETA_MAX = R["theta"][1]
U = lambda s: rf"\;(\mathrm{{{s}}})"

def num(label, key, default, step, fmt, help_):
    return st.number_input(
        label, value=dv(R, key, default), step=step,
        min_value=R[key][0], max_value=R[key][1],
        format=fmt if fmt else None, help=help_
    )

# Main layout
left, right = st.columns([1.5, 1], gap="large")

with left:
    st.markdown("""
    <div style="background:transparent; border-radius:12px; padding:0px; margin:-20px 0 0 0; box-shadow:none;">
        <div style="text-align:center; font-size:25px; font-weight:600; color:#333; margin:0; padding:2px;">
            Predict Damage index (DI) for RC Shear Walls
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1px;'></div>" * 3, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin: -80px 0 0 0; padding: 0;">
        <div class='form-banner'>Inputs Features</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1], gap="small")

    with c1:
        st.markdown("<div class='section-header'>Geometry </div>", unsafe_allow_html=True)
        lw = num(rf"$l_w{U('mm')}$","lw",1000.0,1.0,None,"Length")
        hw = num(rf"$h_w{U('mm')}$","hw",495.0,1.0,None,"Height")
        tw = num(rf"$t_w{U('mm')}$","tw",200.0,1.0,None,"Thickness")
        b0 = num(rf"$b_0{U('mm')}$","b0",200.0,1.0,None,"Boundary element width")
        db = num(rf"$d_b{U('mm')}$","db",400.0,1.0,None,"Boundary element length")
        AR = num(r"$AR$","AR",2.0,0.01,None,"Aspect ratio")
        M_Vlw = num(r"$M/(V_{l_w})$","M_Vlw",2.0,0.01,None,"Shear span ratio")

    with c2:
        st.markdown("<div class='section-header'>Reinf. Ratios </div>", unsafe_allow_html=True)
        rt = num(r"$\rho_t\;(\%)$","rt",0.25,0.0001,"%.6f","Transverse web ratio")
        rsh = num(r"$\rho_{sh}\;(\%)$","rsh",0.25,0.0001,"%.6f","Transverse boundary ratio")
        rl = num(r"$\rho_l\;(\%)$","rl",0.25,0.0001,"%.6f","Vertical web ratio")
        rbl = num(r"$\rho_{bl}\;(\%)$","rbl",0.25,0.0001,"%.6f","Vertical boundary ratio")
        s_db = num(r"$s/d_b$","s_db",0.25,0.01,None,"Hoop spacing ratio")
        axial = num(r"$P/(A_g f'_c)$","axial",0.10,0.001,None,"Axial Load Ratio")
        theta = num(r"$\theta\;(\%)$","theta",THETA_MAX,0.0005,None,"Drift Ratio")

    with c3:
        st.markdown("<div class='section-header'>Material Strengths</div>", unsafe_allow_html=True)
        fc = num(rf"$f'_c{U('MPa')}$","fc",40.0,0.1,None,"Concrete strength")
        fyt = num(rf"$f_{{yt}}{U('MPa')}$","fyt",400.0,1.0,None,"Transverse web yield strength")
        fysh = num(rf"$f_{{ysh}}{U('MPa')}$","fysh",400.0,1.0,None,"Transverse boundary yield strength")
        fyl = num(rf"$f_{{yl}}{U('MPa')}$","fyl",400.0,1.0,None,"Vertical web yield strength")
        fybl = num(rf"$f_{{ybl}}{U('MPa')}$","fybl",400.0,1.0,None,"Vertical boundary yield strength")

    st.markdown("</div></div>", unsafe_allow_html=True)

with right:
    # Hero image
    st.markdown(f"<div style='height:50px'></div>", unsafe_allow_html=True)
    try:
        hero_b64 = b64(BASE_DIR / "logo2-01.png")
        st.markdown(f'<div style="position:relative; left:100px; top:-0px; text-align:left;"><img src="data:image/png;base64,{hero_b64}" width="400"/></div>', unsafe_allow_html=True)
    except: pass

    # Model selection and buttons
    col1, col2 = st.columns([3, 1])
    
    with col2:
        available = set(model_registry.keys())
        order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
        ordered_keys = [m for m in order if m in available] or ["(no models loaded)"]
        display_labels = ["RF" if m == "Random Forest" else m for m in ordered_keys]
        _label_to_key = {"RF": "Random Forest"}
        model_choice_label = st.selectbox("Model Selection", display_labels, key="model_select_compact")
        model_choice = _label_to_key.get(model_choice_label, model_choice_label)

        submit = st.button("Calculate", key="calc_btn", use_container_width=True)
        if st.button("Reset", key="reset_btn", use_container_width=True):
            st.rerun()
        if st.button("Clear All", key="clear_btn", use_container_width=True):
            st.session_state.results_df = pd.DataFrame()
        
        # Prediction display
        if not st.session_state.results_df.empty:
            latest_pred = st.session_state.results_df.iloc[-1]["Predicted_DI"]
            st.markdown(f"<div class='prediction-with-color'>Predicted Damage Index (DI): {latest_pred:.4f}</div>", unsafe_allow_html=True)
            
            # Download button
            csv = st.session_state.results_df.to_csv(index=False)
            st.download_button("📂 Download as CSV", data=csv, file_name="di_predictions.csv", 
                              mime="text/csv", use_container_width=True, key="dl_csv_main")

    # Chart slot - MOVED UP
    chart_slot = st.empty()

# Prediction engine
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
        prediction = float(model_registry["LightGBM"].predict(X)[0])
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

def render_di_chart(results_df: pd.DataFrame, curve_df: pd.DataFrame, theta_max: float = THETA_MAX, di_max: float = 1.5, size: int = 460):
    import altair as alt
    selection = alt.selection_point(name='select', fields=['θ', 'Predicted_DI'], nearest=True, on='mouseover', empty=False, clear='mouseout')
    
    base_axes_df = pd.DataFrame({"θ": [0.0, theta_max], "Predicted_DI": [0.0, 0.0]})
    x_ticks = np.linspace(0.0, theta_max, 5).round(2)

    axes_layer = (
        alt.Chart(base_axes_df).mark_line(opacity=0).encode(
            x=alt.X("θ:Q", title="Drift Ratio (θ)", scale=alt.Scale(domain=[0, theta_max], nice=False, clamp=True),
                    axis=alt.Axis(values=list(x_ticks), labelFontSize=14, titleFontSize=16)),
            y=alt.Y("Predicted_DI:Q", title="Damage Index (DI)", scale=alt.Scale(domain=[0, di_max], nice=False, clamp=True),
                    axis=alt.Axis(values=[0.0, 0.2, 0.5, 1.0, 1.5], labelFontSize=14, titleFontSize=16)),
        ).properties(width=size, height=size)
    )

    curve = curve_df if (curve_df is not None and not curve_df.empty) else pd.DataFrame({"θ": [], "Predicted_DI": []})
    line_layer = alt.Chart(curve).mark_line(strokeWidth=2).encode(x="θ:Q", y="Predicted_DI:Q")

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
             .configure_axis(domain=True, ticks=True))
    chart_html = chart.to_html()
    st.components.v1.html(chart_html, height=size + 100)

# Prediction execution
_order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
_label_to_key = {"RF": "Random Forest"}

def _pick_default_model():
    for m in _order:
        if m in model_registry:
            return m
    return None

if 'model_choice' not in locals():
    _label = st.session_state.get("model_select_compact")
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
        except Exception as e:
            st.error(f"Prediction failed for {model_choice}: {e}")

    _base_xdf = _make_input_df(lw, hw, tw, fc, fyt, fysh, fyl, fybl, rt, rsh, rl, rbl, axial, b0, db, s_db, AR, M_Vlw, theta)
    _curve_df = _sweep_curve_df(model_choice, _base_xdf, theta_max=THETA_MAX, step=0.1)

# Render chart - MOVED UP
with right:
    with chart_slot:
        render_di_chart(st.session_state.results_df, _curve_df, theta_max=THETA_MAX, di_max=1.5, size=400)
