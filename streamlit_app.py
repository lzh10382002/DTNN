"""Streamlit Community Cloud entrypoint for the DTNN RF predictor."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import streamlit as st

from dtnn_web_model import DTNNModel


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

MODEL_1_DIR = MODELS_DIR / "rf_q_model_1pct"
MODEL_2_DIR = MODELS_DIR / "rf_q_model_2pct"


st.set_page_config(
    page_title="RF & Q-RF Prediction System",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the 1% and 2% DTNN models...")
def load_dtnn_models() -> tuple[DTNNModel, DTNNModel]:
    """Load both original TensorFlow checkpoints once per app process."""

    return DTNNModel(MODEL_1_DIR), DTNNModel(MODEL_2_DIR)


def asset_path(filename: str) -> Path | None:
    path = ASSETS_DIR / filename
    return path if path.is_file() else None


def determine_support_category(q_value: float, span: float) -> tuple[str, str]:
    """Apply the same Q-system support boundaries as the desktop application."""

    x = float(q_value)
    y = float(span)

    if y >= 4 and x <= 100 and y < (11 * x / 96 + 85 / 24):
        return "Category ①", "Unsupported or spot bolting"
    if y <= 30 and x <= 100 and y >= (11 * x / 96 + 85 / 24) and y < (0.44 * x - 5.2):
        return "Category ②", "Spot bolting"
    if (
        4 <= y <= 30
        and y >= (0.44 * x - 5.2)
        and y >= (11 * x / 96 + 85 / 24)
        and y < (13 * x / 7 + 15 / 7)
    ):
        return "Category ③", "Systematic bolting + Fibre reinforced sprayed concrete (5–6 cm)"
    if 4 <= y <= 30 and y >= (13 * x / 7 + 15 / 7) and y < (104 * x / 23 + 66 / 23):
        return "Category ④", "Fibre reinforced sprayed concrete (6–9 cm) + Bolting"
    if 4 <= y <= 30 and y >= (104 * x / 23 + 66 / 23) and y < (2600 * x / 143 + 390 / 143):
        return "Category ⑤", "Fibre reinforced sprayed concrete (9–12 cm) + Bolting"
    if 4 <= y <= 30 and y >= (2600 * x / 143 + 390 / 143) and y < (13000 * x / 241 + 730 / 241):
        return (
            "Category ⑥",
            "Fibre reinforced sprayed concrete (12–15 cm) + Reinforced ribs of sprayed concrete and bolting (RRS-A) + Bolting",
        )
    if 4 <= y <= 30 and y >= (13000 * x / 241 + 730 / 241) and y < (26000 * x / 179 + 690 / 179):
        return (
            "Category ⑦",
            "Fibre reinforced sprayed concrete (>15 cm) + Reinforced ribs of sprayed concrete and bolting (RRS-B) + Bolting",
        )
    if 4 <= y <= 30 and y >= (26000 * x / 179 + 690 / 179) and y < (5000 * x / 33 + 490 / 33):
        return (
            "Category ⑧",
            "Fibre reinforced sprayed concrete (>25 cm) + Double layer ribs of sprayed concrete and bolting (RRS-C) + Bolting",
        )
    if 4 <= y <= 30 and x >= 0.001 and y >= (5000 * x / 33 + 490 / 33):
        return "Category ⑨", "Special evaluation required"

    return "Unknown Category", "Parameters outside the defined support regions"


def result_card(title: str, value: str, color: str, background: str = "#e1e1e1") -> None:
    safe_title = html.escape(title)
    safe_value = html.escape(value)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-title" style="background:{background};">{safe_title}</div>
            <div class="result-value" style="color:{color};">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        .result-card {border:1px solid #c9c9c9; border-radius:6px; text-align:center; margin-bottom:12px; min-height:110px; overflow:hidden;}
        .result-title {font-weight:700; padding:7px;}
        .result-value {font-size:1.8rem; font-weight:700; padding:20px 10px;}
        .support-value {font-size:1.05rem; font-weight:700; color:#d2691e; margin:12px 8px 5px;}
        .support-description {font-size:0.9rem; color:#555; padding:0 12px 12px;}
        .anchor-value {font-size:1.05rem; margin:9px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    model_1pct, model_2pct = load_dtnn_models()
except Exception as exc:
    st.error("The DTNN models could not be loaded.")
    st.exception(exc)
    st.stop()


left_column, right_column = st.columns([4.5, 5.5], gap="large")

with left_column:
    q_chart = asset_path("Q value.png")
    if q_chart:
        st.image(q_chart, width="stretch", caption="Q-Value Chart")
    else:
        st.info("Q-Value chart not found.")

    st.divider()

    support_chart = asset_path("Support Method.png")
    if support_chart:
        st.image(support_chart, width="stretch", caption="Support Method Chart")
    else:
        st.info("Support Method chart not found.")

with right_column:
    logo_left, logo_right = st.columns(2)
    monash_logo = asset_path("Monash.png")
    seu_logo = asset_path("SEU.png")
    with logo_left:
        if monash_logo:
            st.image(monash_logo, width=220)
    with logo_right:
        if seu_logo:
            st.image(seu_logo, width=220)

    st.title("RF & Q-RF Prediction System")
    st.caption("TensorFlow DTNN · Original 1% and 2% checkpoints")
    st.divider()

    st.subheader("Step 1: Select Model")
    selected_model = st.selectbox(
        "Choose Model Configuration",
        options=("2% model", "1% model"),
        index=0,
    )
    st.success("Both TensorFlow neural-network models are loaded and ready.")

    with st.expander("Loaded model details"):
        st.write(f"1% architecture: `{model_1pct.layers}`")
        st.write(f"2% architecture: `{model_2pct.layers}`")
        st.write("Runtime: TensorFlow compatibility graph on CPU")

    st.divider()
    st.subheader("Step 2: Input Parameters")

    depth = st.slider("Depth (m)", min_value=40, max_value=240, value=100, step=1)
    magnitude = st.slider("Magnitude, Mw", min_value=1, max_value=8, value=6, step=1)
    q_input = st.slider("Q-value", min_value=4, max_value=100, value=50, step=1)
    span = st.slider("Span (m)", min_value=4, max_value=30, value=15, step=1)

    run_prediction = st.button("🚀 Run Prediction", type="primary", width="stretch")

    st.divider()
    st.subheader("Step 3: Results")

    if run_prediction:
        try:
            log_q = float(np.log10(q_input))
            rf_1pct = model_1pct.predict_one(depth, magnitude, span, log_q)
            rf_2pct = model_2pct.predict_one(depth, magnitude, span, log_q)

            if selected_model == "2% model":
                raw_rf = rf_2pct
                decision_log = f"2% model RF={rf_2pct:.6f}"
            else:
                raw_rf = min(rf_1pct, rf_2pct)
                selected_source = "1%" if rf_1pct <= rf_2pct else "2% (conservative cap)"
                decision_log = (
                    f"1% RF={rf_1pct:.6f}; 2% RF={rf_2pct:.6f}; "
                    f"using {selected_source}"
                )

            rf_value = float(np.clip(raw_rf, 0.0, 1.0))

            if rf_value <= 0.0:
                adjusted_q_text = "NaN"
                category_title = "Category ⑨"
                category_description = "Special evaluation required"
                anchor_spacing_text = "NaN"
                anchor_length_text = "NaN"
            else:
                adjusted_q = q_input * rf_value
                adjusted_q_text = f"{adjusted_q:.4f}"
                category_title, category_description = determine_support_category(adjusted_q, span)
                anchor_spacing = 10 ** (0.24 + 0.12 * np.log10(adjusted_q))
                anchor_length = 1.4 + 0.184 * span
                anchor_spacing_text = f"{anchor_spacing:.2f} m"
                anchor_length_text = f"{anchor_length:.2f} m"

            result_left, result_right = st.columns(2)
            with result_left:
                result_card("RF (Neural Net)", f"{rf_value:.4f}", "#0055aa")
            with result_right:
                result_card("Calculated Q (Q × RF)", adjusted_q_text, "#228b22")

            support_column, anchor_column = st.columns(2)
            with support_column:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title" style="background:#ffebcd;">Recommended Support Category</div>
                        <div class="support-value">{html.escape(category_title)}</div>
                        <div class="support-description">{html.escape(category_description)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with anchor_column:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title" style="background:#e0ffff;">Anchor Length & Spacing</div>
                        <div class="anchor-value"><b>Length (L):</b> {html.escape(anchor_length_text)}</div>
                        <div class="anchor-value"><b>Spacing (s):</b> {html.escape(anchor_spacing_text)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(f"Model decision: {decision_log}")
        except Exception as exc:
            st.error("Prediction failed.")
            st.exception(exc)
    else:
        st.info("Adjust the parameters and click Run Prediction.")

