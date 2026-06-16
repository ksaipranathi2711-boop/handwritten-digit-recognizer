"""
app.py
-------
Streamlit web app for Handwritten Digit Recognition using a CNN
trained on the MNIST dataset.

Features:
    - Draw a digit on an interactive canvas
    - Upload an image (JPG/JPEG/PNG)
    - Predict digit + confidence + probability distribution
    - Prediction history
    - Modern SaaS-style UI
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import plotly.express as px

from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom CSS - modern AI SaaS look
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    .hero {
        background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(168,85,247,0.12));
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    .hero h1 {
        font-size: 2.4rem;
        margin: 0 0 .4rem 0;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .hero p { color: #cbd5e1; font-size: 1.05rem; margin: 0; }
    .card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 1.25rem 1.4rem;
        border-radius: 14px;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    .metric-big {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .metric-label {
        text-align: center;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: .8rem;
        margin-top: .4rem;
    }
    .confidence {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #34d399;
        margin-top: .3rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        border: none;
        padding: .65rem 1.4rem;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(99,102,241,.35);
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 1.2rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(148,163,184,.15);
        font-size: .9rem;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.85);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------
MODEL_PATH = "digit_model.h5"


@st.cache_resource(show_spinner=False)
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


model = get_model()

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 Project Info")
    st.markdown(
        """
**Handwritten Digit Recognizer**
A deep learning app powered by a Convolutional Neural Network trained
on the MNIST dataset.
        """
    )

    with st.expander("📐 CNN Architecture", expanded=False):
        st.code(
            """Conv2D(32, 3x3, relu)
Conv2D(64, 3x3, relu)
MaxPooling2D(2x2) + Dropout(0.25)
Conv2D(128, 3x3, relu)
MaxPooling2D(2x2) + Dropout(0.25)
Flatten
Dense(128, relu) + Dropout(0.5)
Dense(10, softmax)""",
            language="text",
        )

    with st.expander("📊 Dataset"):
        st.markdown(
            """
- **Dataset:** MNIST
- **Train samples:** 60,000
- **Test samples:** 10,000
- **Image size:** 28x28 grayscale
- **Classes:** 0–9
            """
        )

    with st.expander("🎯 Model Performance"):
        st.metric("Test Accuracy", "~99.2%")
        st.metric("Test Loss", "~0.03")

    with st.expander("🛠 Technologies"):
        st.markdown(
            "- Python\n- TensorFlow / Keras\n- Streamlit\n- NumPy / Pandas\n"
            "- Pillow\n- Plotly\n- streamlit-drawable-canvas"
        )

    with st.expander("👨‍💻 Developer"):
        st.markdown(
            "Built as an **AI/ML Internship** project.\n\n"
            "Trained on MNIST using a custom CNN."
        )

    st.markdown("---")
    st.caption("v1.0 • Built with ❤️ using Streamlit")

# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <h1>🧠 Handwritten Digit Recognizer</h1>
    <p>Recognize handwritten digits using Deep Learning and Convolutional Neural Networks.</p>
</div>
""",
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "⚠️ Model file `digit_model.h5` not found. Run `python train_model.py` first to train the model."
    )
    st.stop()

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Convert any input PIL image to MNIST-style (1, 28, 28, 1) tensor."""
    img = pil_img.convert("L")  # grayscale

    arr = np.array(img)
    # If background is light, invert so digit is white on black (MNIST style)
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    # Crop to bounding box of the digit (non-zero pixels)
    arr = np.array(img)
    coords = np.argwhere(arr > 30)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img.crop((x0, y0, x1, y1))

    # Resize the digit so its largest side is 20px, then pad to 28x28
    img.thumbnail((20, 20), Image.LANCZOS)
    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(
        img,
        ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2),
    )

    arr = np.array(new_img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


def predict(pil_img: Image.Image):
    x = preprocess_image(pil_img)
    probs = model.predict(x, verbose=0)[0]
    return int(np.argmax(probs)), float(np.max(probs)) * 100, probs


# ---------------------------------------------------------------------
# Input tabs
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["✏️ Draw a Digit", "📤 Upload Image"])

input_image: Image.Image | None = None

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
    with col_b:
        st.markdown("**Instructions**")
        st.markdown(
            "- Draw a single digit (0–9)\n"
            "- Use the toolbar's 🗑 icon to clear\n"
            "- Then click **Predict Digit**"
        )

    if canvas_result.image_data is not None:
        arr = canvas_result.image_data.astype("uint8")
        if arr[:, :, :3].sum() > 0:
            input_image = Image.fromarray(arr)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload an image of a handwritten digit",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        input_image = Image.open(uploaded)
        st.image(input_image, caption="Uploaded Image", width=200)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------
st.markdown("### ")
predict_btn = st.button("🚀 Predict Digit", use_container_width=True)

if predict_btn:
    if input_image is None:
        st.warning("Please draw a digit or upload an image first.")
    else:
        with st.spinner("🔎 Analyzing image..."):
            digit, confidence, probs = predict(input_image)

        st.session_state.history.append(
            {"Digit": digit, "Confidence (%)": round(confidence, 2)}
        )

        # Result cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-big">{digit}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Predicted Digit</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="confidence">{confidence:.2f}%</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
            st.progress(min(int(confidence), 100))
            st.markdown("</div>", unsafe_allow_html=True)

        # Probability chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Prediction Probability Distribution")
        df = pd.DataFrame({"Digit": list(range(10)), "Probability": probs * 100})
        colors = ["#a855f7" if i == digit else "#475569" for i in range(10)]
        fig = px.bar(df, x="Digit", y="Probability", text="Probability")
        fig.update_traces(
            marker_color=colors,
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            yaxis=dict(range=[0, 110], title="Probability (%)"),
            xaxis=dict(dtick=1),
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# History
# ---------------------------------------------------------------------
if st.session_state.history:
    with st.expander("📜 Prediction History", expanded=False):
        st.dataframe(
            pd.DataFrame(st.session_state.history[::-1]),
            use_container_width=True,
        )
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    """
<div class="footer">
    🧠 Handwritten Digit Recognizer • Built with TensorFlow + Streamlit • AI/ML Internship Project
</div>
""",
    unsafe_allow_html=True,
)
