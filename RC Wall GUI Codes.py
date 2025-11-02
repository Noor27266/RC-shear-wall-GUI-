# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

# ML
import xgboost as xgb
import joblib
import catboost
import lightgbm as lgb
from tensorflow.keras.models import load_model

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

def css(s: str): st.markdown(s, unsafe_allow_html=True)
def b64(p: Path) -> str:
    try: return base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception: return ""

# -----------------------------------------------------------------------------
# Global responsive CSS (auto-scales with viewport)
# -----------------------------------------------------------------------------
css("""
<style>
  /* Center the app and let it breathe on any screen */
  .block-container{
    padding-top:.6rem;
    max-width: min(1400px, 96vw);
    margin-left:auto; margin-right:auto;
  }

  /* Responsive font sizes using clamp(min, vw-based, max) */
  h1.app-title{
    font-size: clamp(24px, 3.2vw, 44px);
    font-weight: 800; text-align:center; margin:.4rem 0 1rem 0 !important;
  }

  .form-banner{
    font-size: clamp(18px, 2.2vw, 30px);
    font-weight: 800;
    background: linear-gradient(90deg,#0E9F6E,#84CC16);
    color:#fff; padding:.45rem .9rem; border-radius:12px; text-align:center;
    margin:.2rem 0 .8rem 0 !important;
  }

  .section-header{
    font-size: clamp(16px, 1.6vw, 26px);
    font-weight: 700; margin:.25rem 0 .5rem 0;
  }

  /* Inputs scale with viewport */
  .stNumberInput label, .stSelectbox label{
    font-size: clamp(13px, 1.2vw, 20px) !important; font-weight:700;
  }
  .stNumberInput label .katex, .stSelectbox label .katex{
    font-size: clamp(13px, 1.2vw, 20px) !important; line-height:1.2 !important;
  }
  .stNumberInput label .katex .mathrm,
  .stSelectbox  label .katex .mathrm{ font-size: clamp(11px, .9vw, 16px) !important; }

  /* Input boxes */
  div[data-testid="stNumberInput"] input[type="number"],
  div[data-testid="stNumberInput"] input[type="text"]{
      font-size: clamp(12px, 1.1vw, 18px) !important;
      height: clamp(32px, 4.2vh, 56px) !important;
      line-height: 1.1 !important;
      font-weight:600 !important;
      padding:10px 12px !important;
  }
  div[data-testid="stNumberInput"] [data-baseweb*="input"]{
      background:#fff !important;
      border:1px solid #e6e9f2 !important;
      border-radius:10px !important;
      box-shadow:0 1px 2px rgba(16,24,40,.06) !important;
  }

  /* Buttons */
  div.stButton > button{
    font-size: clamp(12px, 1.1vw, 18px) !important;
    height: clamp(34px, 5vh, 48px) !important;
    color:#fff !important; font-weight:700; border:none !important;
    border-radius:10px !important; white-space:nowrap !important;
    padding: 0 .9rem !important; background:#4CAF50 !important;
  }
  button[key="reset_btn"]{ background:#2196F3 !important; }
  button[key="clear_btn"]{ background:#f44336 !important; }

  /* Prediction badge */
  .prediction-result{
    font-size: clamp(12px, 1.05vw, 17px) !important; font-weight:700; color:#2e86ab;
    background:#f1f3f4; padding:.5rem .8rem; border-radius:10px;
    text-align:center; display:inline-block; white-space:nowrap;
  }
</style>
""")

# -----------------------------------------------------------------------------
# Sidebar (minimal; no UI scale)
# -----------------------------------------------------------------------------
with st.sidebar:
    right_offset = st.slider("Right panel vertical offset (px)", -60, 200, 16, 2)
    show_logo = st.checkbox("Show header logo", True)
    logo_px = st.slider("Logo size (px)", 28, 160, 80, 2)

# Header
logo_html = ""
if show_logo:
    p = Path(__file__).resolve().parent / "TJU logo.png"
    _b64 = b64(p)
    if _b64:
        logo_html = f'<img src="data:image/png;base64,{_b64}" style="height:{int(logo_px)}px; vertical-align:middle; margin-left:.6rem" alt="Logo"/>'
st.markdown(f'<h1 class="app-title">Predict Damage index (DI) for RC Shear Walls {logo_html}</h1>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load models (same logic, tolerant)
# -----------------------------------------------------------------------------
def record_health(name, ok, msg=""): health.append((name, ok, msg))
health = []
class _ScalerShim:
    def __init__(self, Xs, Ys): import numpy as _np; self._np=_np; self.Xs=Xs; self.Ys=Ys
    def transform_X(self, X): return self.Xs.transform(X)
    def inverse_transform_y(self, y): y=self._np.array(y).reshape(-1,1); return self.Ys.inverse_transform(y)

model_registry = {}

try:
    ann_ps_model = load_model("ANN_PS_Model.keras")
    ann_ps_proc  = _ScalerShim(joblib.load("ANN_PS_Scaler_X.save"), joblib.load("ANN_PS_Scaler_y.save"))
    model_registry["PS"] = ann_ps_model; record_health("PS (ANN)", True, "loaded")
except Exception as e: record_health("PS (ANN)", False, str(e))

try:
    ann_mlp_model = load_model("ANN_MLP_Model.keras")
    ann_mlp_proc  = _ScalerShim(joblib.load("ANN_MLP_Scaler_X.save"), joblib.load("ANN_MLP_Scaler_y.save"))
    model_registry["MLP"] = ann_mlp_model; record_health("MLP (ANN)", True, "loaded")
except Exception as e: record_health("MLP (ANN)", False, str(e))

try:
    rf_model = joblib.load("random_forest_model.pkl")
    model_registry["Random Forest"] = rf_model; record_health("Random Forest", True, "loaded")
except Exception as e: record_health("Random Forest", False, str(e))

try:
    xgb_model = xgb.XGBRegressor(); xgb_model.load_model("XGBoost_trained_model_for_DI.json")
    model_registry["XGBoost"] = xgb_model; record_health("XGBoost", True, "loaded")
except Exception as e: record_health("XGBoost", False, str(e))

try:
    cat_model = catboost.CatBoostRegressor(); cat_model.load_model("CatBoost.cbm")
    model_registry["CatBoost"] = cat_model; record_health("CatBoost", True, "loaded")
except Exception as e: record_health("CatBoost", False, str(e))

def load_lightgbm_flex():
    for p in ["LightGBM_model","LightGBM_model.txt","LightGBM_model.bin","LightGBM_model.pkl","LightGBM_model.joblib"]:
        if Path(p).exists():
            try: return lgb.Booster(model_file=p)
            except Exception:
                try: return joblib.load(p)
                except Exception: pass
    return None

lgb_model = load_lightgbm_flex()
if lgb_model is not None:
    model_registry["LightGBM"] = lgb_model; record_health("LightGBM", True, "loaded")
else:
    record_health("LightGBM", False, "not found")

with st.sidebar:
    st.header("Model Health")
    for name, ok, msg in health:
        st.markdown(f"- {'✅' if ok else '❌'} **{name}**<br/><small>{msg}</small>", unsafe_allow_html=True)

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

# -----------------------------------------------------------------------------
# Inputs (unchanged logic)
# -----------------------------------------------------------------------------
R = {
    "lw":(400.0,3500.0), "hw":(495.0,5486.4), "tw":(26.0,305.0), "fc":(13.38,93.6),
    "fyt":(0.0,1187.0), "fysh":(0.0,1375.0), "fyl":(160.0,1000.0), "fybl":(0.0,900.0),
    "rt":(0.000545,0.025139), "rsh":(0.0,0.041888), "rl":(0.0,0.029089), "rbl":(0.0,0.031438),
    "axial":(0.0,0.86), "b0":(45.0,3045.0), "db":(0.0,500.0), "s_db":(0.0,47.65625),
    "AR":(0.388889,5.833333), "M_Vlw":(0.388889,4.1), "theta":(0.0275,4.85),
}
THETA_MAX = R["theta"][1]
U = lambda s: rf"\\;(\\mathrm{{{s}}})"
dv = lambda R,k,v: float(max(R[k][0], min(v, R[k][1])))

def num(label, key, default, step, fmt, help_):
    return st.number_input(label, value=dv(R,key,default), step=step,
                           min_value=R[key][0], max_value=R[key][1],
                           format=fmt if fmt else None, help=help_)

GEOM=[(rf"$l_w{U('mm')}$","lw",1000.0,1.0,None,"Length"),
      (rf"$h_w{U('mm')}$","hw",495.0,1.0,None,"Height"),
      (rf"$t_w{U('mm')}$","tw",200.0,1.0,None,"Thickness"),
      (rf"$b_0{U('mm')}$","b0",200.0,1.0,None,"Boundary element width"),
      (rf"$d_b{U('mm')}$","db",400.0,1.0,None,"Boundary element length"),
      (r"$AR$","AR",2.0,0.01,None,"Aspect ratio"),
      (r"$M/(V_{l_w})$","M_Vlw",2.0,0.01,None,"Shear span ratio")]
MATS=[(rf"$f'_c{U('MPa')}$","fc",40.0,0.1,None,"Concrete strength"),
      (rf"$f_{{yt}}{U('MPa')}$","fyt",400.0,1.0,None,"Transverse web yield strength"),
      (rf"$f_{{ysh}}{U('MPa')}$","fysh",400.0,1.0,None,"Transverse boundary yield strength"),
      (rf"$f_{{yl}}{U('MPa')}$","fyl",400.0,1.0,None,"Vertical web yield strength"),
      (rf"$f_{{ybl}}{U('MPa')}$","fybl",400.0,1.0,None,"Vertical boundary yield strength")]
REINF=[(r"$\\rho_t\\;(\\%)$","rt",0.25,0.0001,"%.6f","Transverse web ratio"),
       (r"$\\rho_{sh}\\;(\\%)$","rsh",0.25,0.0001,"%.6f","Transverse boundary ratio"),
       (r"$\\rho_l\\;(\\%)$","rl",0.25,0.0001,"%.6f","Vertical web ratio"),
       (r"$\\rho_{bl}\\;(\\%)$","rbl",0.25,0.0001,"%.6f","Vertical boundary ratio"),
       (r"$s/d_b$","s_db",0.25,0.01,None,"Hoop spacing ratio"),
       (r"$P/(A_g f'_c)$","axial",0.10,0.001,None,"Axial Load Ratio"),
       (r"$\\theta\\;(\\%)$","theta",THETA_MAX,0.0005,None,"Drift Ratio")]

left, right = st.columns([1.5, 1.6], gap="large")

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
    st.markdown(f"<div style='height:{int(right_offset)}px'></div>", unsafe_allow_html=True)
    st.image("logo2-01.png", use_column_width=True)  # auto width

    row = st.columns([1.2, 1.2, 1.2], gap="small")
    with row[0]:
        order=["CatBoost","XGBoost","LightGBM","MLP","Random Forest","PS"]
        available=[m for m in order if m in model_registry] or ["(no models loaded)"]
        label_map={"Random Forest":"RF"}; inv={"RF":"Random Forest"}
        display=[label_map.get(m,m) for m in available]
        label = st.selectbox("Model Selection", display, key="model_select")
        model_choice = inv.get(label, label)
    with row[1]:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        submit = st.button("Calculate", key="calc_btn")
    with row[2]:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        if st.button("Reset", key="reset_btn"): st.rerun()
        if st.button("Clear All", key="clear_btn"):
            st.session_state.results_df = pd.DataFrame()
            st.success("All predictions cleared.")

    badge_col, dl_col = st.columns([1.6, 1.0], gap="small")
    with badge_col: pred_banner = st.empty()
    with dl_col: dl_slot = st.empty()

    chart_slot = st.empty()  # chart fills column width

# -----------------------------------------------------------------------------
# Predict + chart (same logic, chart is responsive)
# -----------------------------------------------------------------------------
_TRAIN_NAME_MAP={'l_w':'lw','h_w':'hw','t_w':'tw','f′c':'fc','fyt':'fyt','fysh':'fysh','fyl':'fyl','fybl':'fybl',
                 'ρt':'pt','ρsh':'psh','ρl':'pl','ρbl':'pbl','P/(Agf′c)':'P/(Agfc)','b0':'b0','db':'db','s/db':'s/db',
                 'AR':'AR','M/Vlw':'M/Vlw','θ':'θ'}
_TRAIN_COL_ORDER=['lw','hw','tw','fc','fyt','fysh','fyl','fybl','pt','psh','pl','pbl','P/(Agfc)','b0','db','s/db','AR','M/Vlw','θ']

def _df_in_train_order(df): return df.rename(columns=_TRAIN_NAME_MAP).reindex(columns=_TRAIN_COL_ORDER)

def predict_di(choice, _, input_df):
    df=_df_in_train_order(input_df); X=df.values.astype(np.float32)
    if choice=="LightGBM": y=model_registry["LightGBM"].predict(X)[0]
    elif choice=="XGBoost": y=model_registry["XGBoost"].predict(X)[0]
    elif choice=="CatBoost": y=model_registry["CatBoost"].predict(X)[0]
    elif choice=="Random Forest": y=model_registry["Random Forest"].predict(X)[0]
    elif choice=="PS":
        Xn=ann_ps_proc.transform_X(X); y=ann_ps_model.predict(Xn, verbose=0)[0][0]; y=ann_ps_proc.inverse_transform_y(y).item()
    elif choice=="MLP":
        Xn=ann_mlp_proc.transform_X(X); y=ann_mlp_model.predict(Xn, verbose=0)[0][0]; y=ann_mlp_proc.inverse_transform_y(y).item()
    else: raise RuntimeError("No model available")
    return max(0.035, min(float(y), 1.5))

def _make_df(lw,hw,tw,fc,fyt,fysh,fyl,fybl,rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw,theta):
    cols=['l_w','h_w','t_w','f′c','fyt','fysh','fyl','fybl','ρt','ρsh','ρl','ρbl','P/(Agf′c)','b0','db','s/db','AR','M/Vlw','θ']
    x=np.array([[lw,hw,tw,fc,fyt,fysh,fyl,fybl,rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw,theta]],dtype=np.float32)
    return pd.DataFrame(x, columns=cols)

def _curve_df(model_choice, base_df, theta_max=THETA_MAX, step=0.1):
    if model_choice not in model_registry: return pd.DataFrame(columns=["θ","Predicted_DI"])
    ths=np.round(np.arange(0.0, theta_max+1e-9, step),2)
    rows=[]
    for th in ths:
        df=base_df.copy(); df.loc[:, 'θ']=float(th)
        rows.append({"θ":float(th),"Predicted_DI":float(predict_di(model_choice,None,df))})
    return pd.DataFrame(rows)

def render_chart(curve_df):
    import altair as alt
    chart = (
        alt.Chart(curve_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("θ:Q", title="Drift Ratio (θ)"),
            y=alt.Y("Predicted_DI:Q", title="Damage Index (DI)", scale=alt.Scale(domain=[0,1.5]))
        )
        .properties(width="container", height=400)   # responsive width
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)

order=["CatBoost","XGBoost","LightGBM","MLP","Random Forest","PS"]
model_choice=st.session_state.get("model_select") or next((m for m in order if m in model_registry), None)

if (model_choice is None) or (model_choice not in model_registry):
    st.error("No trained model is available. Please check the Model Selection on the right.")
else:
    xdf=_make_df(lw,hw,tw,fc,fyt,fysh,fyl,fybl,rt,rsh,rl,rbl,axial,b0,db,s_db,AR,M_Vlw,theta)
    if 'submit' not in locals(): submit=False
    if submit:
        try:
            pred=predict_di(model_choice,None,xdf)
            row=xdf.copy(); row["Predicted_DI"]=pred
            st.session_state.results_df=pd.concat([st.session_state.results_df,row], ignore_index=True)
            pred_banner.markdown(f"<div class='prediction-result'>Predicted Damage Index (DI): {pred:.4f}</div>", unsafe_allow_html=True)
            csv=st.session_state.results_df.to_csv(index=False)
            dl_slot.download_button("📂 Download All Results as CSV", data=csv, file_name="di_predictions.csv", mime="text/csv", key="dl_after_submit")
        except Exception as e:
            st.error(f"Prediction failed for {model_choice}: {e}")

    curve=_curve_df(model_choice, xdf, theta_max=THETA_MAX, step=0.1)
    with chart_slot: render_chart(curve)
