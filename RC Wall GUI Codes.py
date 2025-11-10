# =============================================================================
# 🎮 STEP 9: RIGHT PANEL - CONTROLS & INTERACTION ELEMENTS
# =============================================================================
HERO_X, HERO_Y, HERO_W = 100, -10, 300
MODEL_X, MODEL_Y = 100, -2
CHART_W = 300

with right:
    # REMOVE ALL SPACING
    st.markdown("<div style='height: 1px;'></div>", unsafe_allow_html=True)
    
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
    /* MOVE EVERYTHING UP */
    #action-row { 
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        width: 100% !important;
        margin-top: -25px !important;
        margin-bottom: -10px !important;
    }
    
    /* MAKE MODEL SELECTION LABEL BLACK */
    div[data-testid="stSelectbox"] label p { 
        font-size: {FS_LABEL}px !important; 
        color: #000000 !important; /* CHANGED TO BLACK */
        font-weight: bold !important; 
        margin-bottom: 2px !important;
    }
    
    /* MAKE DROPDOWN TEXT BLACK */
    div[data-testid="stSelectbox"] > div > div { 
        height: 50px !important; 
        display: flex !important; 
        align-items: center !important; 
        margin-top: 0px !important;
        border-radius: 8px !important;
        border: none !important;
        outline: none !important;
        color: #000000 !important; /* CHANGED TO BLACK */
    }
    
    /* MAKE DROPDOWN INPUT TEXT BLACK */
    div[data-testid="stSelectbox"] input {
        border: none !important;
        outline: none !important;
        background: transparent !important;
        color: #000000 !important; /* CHANGED TO BLACK */
    }
    
    /* COMPLETELY REMOVE ALL BLACK BORDERS - KEEP GREY BACKGROUND */
    div[data-testid="stSelectbox"] [data-baseweb="select"] {
        border: none !important;
        box-shadow: none !important; 
        background: #D3D3D3 !important;
        height: 50px !important;
        border-radius: 8px !important;
        padding: 0px 12px !important;
        outline: none !important;
    }
    
    div[data-testid="stSelectbox"] > div {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Remove ALL focus borders and black outlines */
    div[data-testid="stSelectbox"] [data-baseweb="select"]:focus,
    div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
    div[data-testid="stSelectbox"] [data-baseweb="select"]:hover {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background-color: #D3D3D3 !important;
    }
    
    /* Remove black from dropdown arrow - MAKE IT DARK GREY */
    div[data-testid="stSelectbox"] svg {
        fill: #666666 !important; /* DARK GREY */
        color: #666666 !important;
        stroke: #666666 !important;
    }
    
    div[data-testid="stSelectbox"] [data-baseweb="select"]:hover svg,
    div[data-testid="stSelectbox"] [data-baseweb="select"]:focus svg {
        fill: #666666 !important;
        color: #666666 !important;
        stroke: #666666 !important;
    }
    
    /* MAKE ENTIRE DROPDOWN OPTIONS BLACK TEXT */
    [data-baseweb="select"] *, 
    [data-baseweb="popover"] *, 
    [data-baseweb="menu"] * { 
        color: #000000 !important; /* CHANGED TO BLACK */
        background-color: #D3D3D3 !important;
        font-size: {FS_SELECT}px !important; 
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    [data-baseweb="popover"] {
        border-radius: 8px !important;
        overflow: hidden !important;
        border: none !important;
        box-shadow: none !important;
        background-color: #D3D3D3 !important;
    }
    
    [data-baseweb="menu"] {
        border: none !important;
        border-radius: 8px !important;
        background-color: #D3D3D3 !important;
    }
    
    div[role="option"] { 
        color: #000000 !important; /* CHANGED TO BLACK */
        font-size: {FS_SELECT}px !important; 
        background-color: #D3D3D3 !important;
        padding: 12px 16px !important;
        border: none !important;
    }
    
    div[role="option"]:hover {
        background-color: #B8B8B8 !important;
        color: #000000 !important; /* CHANGED TO BLACK */
        border: none !important;
    }
    
    /* MOVE BUTTONS UP */
    div.stButton > button { 
        height: 50px !important; 
        width: 90% !important;
        display:flex !important; 
        align-items:center !important; 
        justify-content:center !important;
        font-size: {FS_BUTTON}px !important;
        margin: 0 auto !important;
        white-space: nowrap !important;
        margin-top: -5px !important; /* MOVED UP */
        border-radius: 8px !important;
        border: none !important;
        font-weight: 700 !important;
        outline: none !important;
    }
    
    button[key="calc_btn"] { background:#4CAF50 !important; }
    button[key="reset_btn"] { background:#2196F3 !important; }
    button[key="clear_btn"] { background:#f44336 !important; }
    
    div.stButton > button:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* FIX DOWNLOAD BUTTON - MAKE IT STRAIGHT LINE */
    .stDownloadButton > button {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        white-space: nowrap !important;
        margin: 0 auto !important;
    }
    
    /* FIX PREDICTION RESULT POSITION */
    .prediction-result {
        margin-top: -10px !important;
        margin-bottom: -5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # SINGLE ROW WITH CUSTOM WIDTHS
    st.markdown("<div id='action-row'>", unsafe_allow_html=True)

    model_col, calc_col, reset_col, clear_col = st.columns([1.5, 1, 1, 1], gap="small")

    with model_col:
        available = set(model_registry.keys())
        order = ["CatBoost", "XGBoost", "LightGBM", "MLP", "Random Forest", "PS"]
        ordered_keys = [m for m in order if m in available] or ["(no models loaded)"]
        display_labels = ["RF" if m == "Random Forest" else m for m in ordered_keys]
        _label_to_key = {"RF": "Random Forest"}
        model_choice_label = st.selectbox("Model Selection", display_labels, key="model_select_compact")
        model_choice = _label_to_key.get(model_choice_label, model_choice_label)

    with calc_col:
        submit = st.button("Calculate", key="calc_btn", use_container_width=True)

    with reset_col:
        if st.button("Reset", key="reset_btn", use_container_width=True):
            st.rerun()

    with clear_col:
        if st.button("Clear All", key="clear_btn", use_container_width=True):
            st.session_state.results_df = pd.DataFrame()
            st.success("All predictions cleared.")

    st.markdown("</div>", unsafe_allow_html=True)

    # FIX DOWNLOAD BUTTON LAYOUT
    st.markdown("""
    <style>
    /* FIX THE DOWNLOAD BUTTON COLUMNS */
    [data-testid="column"] {
        min-width: auto !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # USE PROPER COLUMNS FOR DOWNLOAD BUTTON
    dl_col1, dl_col2 = st.columns([1, 1])
    with dl_col1:
        pred_banner = st.empty()
    with dl_col2:
        dl_slot = st.empty()
        if not st.session_state.results_df.empty:
            csv = st.session_state.results_df.to_csv(index=False)
            dl_slot.download_button("📂 Download as CSV", data=csv, file_name="di_predictions.csv", mime="text/csv", use_container_width=True)

    chart_slot = st.empty()
