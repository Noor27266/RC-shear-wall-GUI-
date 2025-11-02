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
# Small helpers
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
# Step #2: Page config + COLORS + font knobs
# =============================================================================
st.set_page_config(page_title="RC Shear Wall DI Estimator", layout="wide", page_icon="🧱")

# ====== ONLY FONTS/LOGO KNOBS BELOW (smaller defaults) ======
SCALE_UI = 0.36  # global shrink (pure scaling; lower => smaller). Safe at 100% zoom.

s = lambda v: int(round(v * SCALE_UI))

FS_TITLE   = s(90)  # page title
FS_SECTION = s(60)  # section headers
FS_LABEL   = s(40)  # input & select labels (katex included)
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
    height:{max(42, int(round(FS_BUTTON*1.45)))}px !important;
    line-height:{max(36, int(round(FS_BUTTON*1.15)))}px !important;
    white-space:nowrap !important;
    color:#fff !important;
    font-weight:700; border:none !important; border-radius:8px !important;
    background:#4CAF50 !important;
  }}
  div.stButton > button:hover {{ filter: brightness(0.95); }}

  button[key="calc_btn"] {{ background:#4CAF50 !important; }}
  button[key="reset_btn"] {{ background:#2196F3 !important; }}
  button[key="clear_btn"] {{ background:#f44336 !important; }}

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

  [data-baseweb="popover"], [data-baseweb="tooltip"],
  [data-baseweb="popover"] > div, [data-baseweb="tooltip"] > div {{
      background:#000 !important; color:#fff !important; border-radius:8px !important;
      padding:6px 10px !important; font-size:{max(14, FS_SELECT)}px !important; font-weight:500 !important;
  }}
  [data-baseweb="popover"] *, [data-baseweb="tooltip"] * {{ color:#fff !important; }}

  /* Keep consistent sizes for model select label and buttons */
  label[for="model_select_compact"] {{ font-size:{FS_LABEL}px !important; font-weight:bold !important; }}
  #action-row {{ display:flex; align-items:center; gap:10px; }}
</style>
""")

# Keep header area slim
st.markdown("""
<style>
html, body{ margin:0 !important; padding:0 !important; }
header[data-testid="stHeader"]{ height:0 !important; padding:0 !important; background:transparent !important; }
header[data-testid="stHeader"] *{ display:none !important; }
div.stApp{ margin-top:-4rem !important; }
section.main > div.block-container{ padding-top:0 !important; margin-top:0 !important; }
/* Keep Altair responsive */
.vega-embed, .vega-embed .chart-wrapper{ max-width:100% !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Hide Streamlit's small +/- buttons on number inputs */
div[data-testid="stNumberInput"] button { display: none !important; }

/* Also hide browser numeric spinners for consistency */
div[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
div[data-testid="stNumberInput"] input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
div[data-testid="stNumberInput"] input[type=number] { -moz-appearance: textfield; }
</style>
""", unsafe_allow_html=True)

# Adjust the DI badge and Download as CSV button spacing:
st.markdown("""
<style>
/* Make the DI badge fill its column (same width as the button when columns are equal) */
.prediction-result{
  width: 100% !important;
  display: block !important;
  margin-right:32px !important;   /* Increased gap */
}

/* Ensure the DI badge stays inline and give it breathing room */
.prediction-result{
  display:inline-flex !important;
  align-items:center !important;
  white-space:nowrap !important;
  width:auto !important;
  margin-right:32px !important;   /* Increased gap */
}

/* Download button */
div[data-testid="stDownloadButton"] button{
  white-space:nowrap !important;   /* single line text */
  display:inline-flex !important;
  align-items:center !important;
  height:auto !important;
  line-height:1.1 !important;
  padding:8px 14px !important;     /* compact; prevents forced wrap */
}

/* Ensure the CSV button itself sits inline */
div[data-testid="stDownloadButton"]{
  display:inline-block !important;
  margin-left:0 !important;        /* keep control of spacing via badge margin */
}
</style>
""", unsafe_allow_html=True)

# Continue with the rest of the code...
