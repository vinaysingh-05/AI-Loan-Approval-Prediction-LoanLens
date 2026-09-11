from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "loan_approval_voting_classifier.joblib"
DATA_PATH = ROOT / "data" / "preprocessed" / "final.csv"

st.set_page_config(
    page_title="LoanLens | Approval Intelligence",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


def inject_styles():
    colors = {
        "background": "#111827",
        "surface": "#182235",
        "surface_alt": "#202d42",
        "text": "#f4f7fb",
        "muted": "#aab7ca",
        "border": "#314158",
        "accent": "#54d6b1",
        "accent_alt": "#7dd3fc",
        "shadow": "rgba(0, 0, 0, .24)",
    }

    st.markdown(
        f"""
        <style>
        :root {{
            --background: {colors['background']};
            --surface: {colors['surface']};
            --surface-alt: {colors['surface_alt']};
            --text: {colors['text']};
            --muted: {colors['muted']};
            --border: {colors['border']};
            --accent: {colors['accent']};
            --accent-alt: {colors['accent_alt']};
            --shadow: {colors['shadow']};
        }}
        .stApp {{ background: var(--background); color: var(--text); }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ display: none; }}
        h1, h2, h3, p, label, [data-testid="stMetricLabel"] {{ color: var(--text) !important; }}
        .muted {{ color: var(--muted); }}
        .hero {{
            position: relative; overflow: hidden; padding: 2.2rem 2.4rem 2.5rem;
            margin-bottom: 1.25rem; border: 1px solid var(--border); border-radius: 22px;
            background: linear-gradient(120deg, var(--surface) 0%, var(--surface-alt) 100%);
            box-shadow: 0 16px 40px var(--shadow); animation: rise-in .65s ease-out both;
        }}
        .hero::after {{
            content: ""; position: absolute; left: -10%; right: -10%; top: 0; height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent), transparent);
            animation: scan 4s ease-in-out infinite;
        }}
        .eyebrow {{ color: var(--accent) !important; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; font-size: .72rem; }}
        .hero h1 {{ margin: .35rem 0 .6rem; font-size: clamp(2rem, 4vw, 3.8rem); line-height: 1; letter-spacing: -.04em; }}
        .hero p {{ max-width: 650px; color: var(--muted) !important; font-size: 1.03rem; margin: 0; }}
        .status-strip {{ display: flex; gap: .65rem; align-items: center; color: var(--muted); font-size: .84rem; margin: .6rem 0 1.2rem; }}
        .status-dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 5px color-mix(in srgb, var(--accent) 18%, transparent); animation: breathe 2s ease-in-out infinite; }}
        .topbar {{ display: flex; align-items: center; gap: 1rem; padding: .75rem 0 1.15rem; border-bottom: 1px solid var(--border); margin-bottom: 1.4rem; }}
        .brand {{ color: var(--text) !important; font-size: 1.2rem; font-weight: 850; letter-spacing: -.03em; margin-right: auto; }}
        .brand-mark {{ color: var(--accent); }}
        .nav-caption {{ color: var(--muted) !important; font-size: .78rem; margin-right: .5rem; }}
        .panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.15rem 1.25rem; box-shadow: 0 10px 28px var(--shadow); animation: rise-in .75s .08s ease-out both; }}
        .section-label {{ font-size: .72rem; font-weight: 800; text-transform: uppercase; letter-spacing: .1em; color: var(--accent) !important; margin-bottom: .55rem; }}
        .result-approved, .result-review {{ padding: 1.35rem; border-radius: 16px; animation: result-in .55s cubic-bezier(.2,.8,.2,1) both; }}
        .result-approved {{ background: linear-gradient(135deg, #0b806c, #155e75); color: white; }}
        .result-review {{ background: linear-gradient(135deg, #aa6b18, #8a3d37); color: white; }}
        .result-approved h2, .result-review h2, .result-approved p, .result-review p {{ color: white !important; }}
        .result-kicker {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; opacity: .78; font-weight: 800; }}
        .result-title {{ font-size: 2rem; font-weight: 800; margin: .25rem 0 .3rem; }}
        .stButton > button {{ border-radius: 10px; border: 1px solid var(--accent); min-height: 2.7rem; font-weight: 750; transition: transform .2s ease, box-shadow .2s ease; }}
        .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 18px var(--shadow); }}
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{ background: var(--surface); border-color: var(--border); }}
        div[data-baseweb="input"] input, div[data-baseweb="select"] * {{ color: var(--text) !important; }}
        @keyframes rise-in {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        @keyframes result-in {{ from {{ opacity: 0; transform: scale(.97) translateY(10px); }} to {{ opacity: 1; transform: scale(1) translateY(0); }} }}
        @keyframes scan {{ 0%, 100% {{ transform: translateX(-35%); opacity: .25; }} 50% {{ transform: translateX(35%); opacity: 1; }} }}
        @keyframes breathe {{ 0%, 100% {{ transform: scale(.85); opacity: .65; }} 50% {{ transform: scale(1.2); opacity: 1; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def confidence_chart(confidence: float, approved: bool):
    color = "#b7f3df" if approved else "#ffd28a"
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 34, "color": color}},
            title={"text": "Model confidence", "font": {"size": 14, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,.6)"},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,.12)",
                "borderwidth": 0,
            },
        )
    )
    figure.update_layout(height=220, margin={"l": 18, "r": 18, "t": 36, "b": 12}, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def main():
    inject_styles()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    st.markdown(
        '<div class="topbar"><div class="brand"><span class="brand-mark">/</span> LoanLens</div><div class="nav-caption">Approval intelligence workspace</div></div>',
        unsafe_allow_html=True,
    )
    nav_home, nav_prediction, nav_configuration = st.columns([1, 1, 1], gap="small")
    with nav_home:
        if st.button("Home", use_container_width=True):
            st.session_state.page = "home"
    with nav_prediction:
        if st.button("Prediction", use_container_width=True):
            st.session_state.page = "prediction"
    with nav_configuration:
        if st.button("Configuration", use_container_width=True):
            st.session_state.page = "configuration"

    page = st.session_state.page

    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Loan decision studio</div>
            <h1>Make the next approval decision clearer.</h1>
            <p>Enter an applicant profile and get a fast, explainable prediction from the trained ensemble model.</p>
        </section>
        <div class="status-strip"><span class="status-dot"></span>Model online <span>•</span> Ready for a new assessment</div>
        """,
        unsafe_allow_html=True,
    )

    if page == "home":
        home_left, home_right = st.columns([1.15, 1], gap="large")
        with home_left:
            st.markdown("### A calmer way to review applications")
            st.markdown('<p class="muted">Use the Prediction workspace to evaluate a profile with the saved ensemble model. The result includes a direct approval status and confidence signal.</p>', unsafe_allow_html=True)
            if st.button("Start a prediction", type="primary"):
                st.session_state.page = "prediction"
                st.rerun()
        with home_right:
            st.markdown('<div class="panel"><div class="section-label">System status</div><h3>Model online</h3><p class="muted">Optimized Voting Classifier</p><p class="muted">5-fold validation accuracy: 98.66%</p></div>', unsafe_allow_html=True)
        return

    if page == "configuration":
        st.markdown("### Configuration")
        config_left, config_right = st.columns(2, gap="large")
        with config_left:
            st.markdown('<div class="panel"><div class="section-label">Model</div><h3>Optimized Voting Classifier</h3><p class="muted">Soft voting across logistic regression, random forest, and LightGBM.</p></div>', unsafe_allow_html=True)
        with config_right:
            st.markdown('<div class="panel"><div class="section-label">Validation</div><h3>98.66% accuracy</h3><p class="muted">Precision 98.57% | Recall 97.89% | F1 98.23%</p></div>', unsafe_allow_html=True)
        st.info("Inputs are scaled with the fitted StandardScaler and passed to the saved model bundle.")
        return

    if not MODEL_PATH.exists():
        st.error("The trained model was not found. Run the final save cell in notebooks/best_model.ipynb first.")
        st.stop()

    bundle = load_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    data = load_data()

    defaults = data[feature_names].median().to_dict() if not data.empty else {
        "no_of_dependents": 2, "education": 1, "self_employed": 0, "income_annum": 5000000,
        "loan_amount": 15000000, "loan_term": 12, "cibil_score": 700,
        "residential_assets_value": 5000000, "commercial_assets_value": 3000000,
        "luxury_assets_value": 5000000, "bank_asset_value": 3000000,
    }

    with st.form("loan_assessment"):
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown('<div class="section-label">Applicant profile</div>', unsafe_allow_html=True)
            dependents = st.number_input("Number of dependents", 0, 20, int(defaults["no_of_dependents"]))
            education_label = st.selectbox("Education", ["Graduate", "Not graduate"], index=int(defaults["education"]))
            employment_label = st.selectbox("Self-employed", ["No", "Yes"], index=int(defaults["self_employed"]))
            cibil_score = st.slider("CIBIL score", 300, 900, int(defaults["cibil_score"]), 1)
            loan_term = st.slider("Loan term (years)", 2, 30, int(defaults["loan_term"]), 1)
        with right:
            st.markdown('<div class="section-label">Financial snapshot</div>', unsafe_allow_html=True)
            income = st.number_input("Annual income", 0.0, 100000000.0, float(defaults["income_annum"]), step=100000.0, format="%.0f")
            loan_amount = st.number_input("Loan amount", 0.0, 100000000.0, float(defaults["loan_amount"]), step=100000.0, format="%.0f")
            residential_assets = st.number_input("Residential assets", 0.0, 100000000.0, float(defaults["residential_assets_value"]), step=100000.0, format="%.0f")
            commercial_assets = st.number_input("Commercial assets", 0.0, 100000000.0, float(defaults["commercial_assets_value"]), step=100000.0, format="%.0f")
            luxury_assets = st.number_input("Luxury assets", 0.0, 100000000.0, float(defaults["luxury_assets_value"]), step=100000.0, format="%.0f")
            bank_assets = st.number_input("Bank assets", 0.0, 100000000.0, float(defaults["bank_asset_value"]), step=100000.0, format="%.0f")

        submitted = st.form_submit_button("Assess application", use_container_width=True, type="primary")

    if submitted:
        values = {
            "no_of_dependents": dependents,
            "education": 0 if education_label == "Graduate" else 1,
            "self_employed": 1 if employment_label == "Yes" else 0,
            "income_annum": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets,
            "commercial_assets_value": commercial_assets,
            "luxury_assets_value": luxury_assets,
            "bank_asset_value": bank_assets,
        }
        input_frame = pd.DataFrame([values], columns=feature_names)
        scaled_input = pd.DataFrame(scaler.transform(input_frame), columns=feature_names)
        prediction = int(model.predict(scaled_input)[0])
        probabilities = model.predict_proba(scaled_input)[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)])
        approved = prediction == 1
        decision_label = "Approved" if approved else "Not Approved"
        st.toast(f"Prediction: {decision_label}")
        if approved:
            st.success(f"Loan application prediction: {decision_label}")
        else:
            st.error(f"Loan application prediction: {decision_label}")

        st.markdown("### Decision signal")
        result_left, result_right = st.columns([1.25, 1], gap="large")
        with result_left:
            result_class = "result-approved" if approved else "result-review"
            title = decision_label
            explanation = "The profile aligns with patterns commonly associated with approved applications." if approved else "The profile differs from the strongest approval patterns; review the application before making a final lending decision."
            st.markdown(
                f'<div class="{result_class}"><div class="result-kicker">Ensemble outcome</div><div class="result-title">{title}</div><p>{explanation}</p></div>',
                unsafe_allow_html=True,
            )
            metric_one, metric_two = st.columns(2)
            metric_one.metric("CIBIL score", f"{cibil_score}")
            metric_two.metric("Loan / income", f"{loan_amount / max(income, 1):.1f}x")
        with result_right:
            confidence_chart(confidence, approved)

        st.caption("This prediction is decision support, not a substitute for responsible lending review or policy checks.")


if __name__ == "__main__":
    main()