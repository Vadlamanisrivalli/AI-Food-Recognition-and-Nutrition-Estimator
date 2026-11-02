"""Streamlit demo: upload or capture a food photo -> dish + calories/macros.

Run:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from foodnutri.config import DEFAULT_CHECKPOINT
from foodnutri.predict import FoodRecognizer

st.set_page_config(page_title="Food Recognition & Nutrition", page_icon="🍽️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_recognizer(path: str) -> FoodRecognizer:
    return FoodRecognizer(path)


st.title("🍽️ AI Food Recognition & Nutrition Estimator")
st.caption("Upload a photo of a meal. A CNN identifies the dish and estimates calories and macros.")

with st.sidebar:
    st.header("Settings")
    ckpt = st.text_input("Model checkpoint", str(DEFAULT_CHECKPOINT))
    topk = st.slider("Top-K predictions", 1, 10, 5)
    portion_mode = st.radio("Portion size", ["Typical serving", "Custom (grams)"])
    custom_grams = st.number_input("Grams", min_value=10, max_value=2000, value=250, step=10,
                                   disabled=portion_mode == "Typical serving")
    weighted = st.checkbox("Blend nutrition across top predictions", value=False,
                           help="Weights each candidate dish by its probability instead of trusting only #1.")
    show_cam = st.checkbox("Show Grad-CAM explanation", value=True)

if not Path(ckpt).exists():
    st.error(f"No trained model found at `{ckpt}`.")
    st.code("python -m foodnutri.train", language="bash")
    st.stop()

recognizer = load_recognizer(ckpt)

tab_upload, tab_camera = st.tabs(["Upload photo", "Use camera"])
with tab_upload:
    uploaded = st.file_uploader("Food image", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    captured = st.camera_input("Take a picture")

source = uploaded if uploaded is not None else captured
if source is None:
    st.info("Add a photo to get started.")
    st.stop()

image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
grams = None if portion_mode == "Typical serving" else float(custom_grams)
result = recognizer.analyze(image, topk=topk, grams=grams, weighted=weighted)
top = result["predictions"][0]
nutrition = result["nutrition"]

left, right = st.columns([1, 1])
with left:
    st.image(image, caption="Your photo")
    if show_cam:
        st.image(recognizer.gradcam(image), caption=f"Grad-CAM: regions that support '{top['label']}'")

with right:
    st.subheader(f"{top['label']}  ·  {top['prob'] * 100:.1f}% confidence")
    if top["prob"] < 0.5:
        st.warning("The model is not very sure about this one. Check the other candidates below.")

    probs = pd.DataFrame({"Dish": [p["label"] for p in result["predictions"]],
                          "Probability": [p["prob"] for p in result["predictions"]]}).set_index("Dish")
    st.bar_chart(probs)

    if nutrition:
        st.markdown(f"#### Estimated nutrition for {nutrition['grams']:.0f} g "
                    f"({'blended' if nutrition['mode'] == 'weighted' else nutrition['label']})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calories", f"{nutrition['kcal']:.0f} kcal")
        c2.metric("Protein", f"{nutrition['protein_g']:.1f} g")
        c3.metric("Carbs", f"{nutrition['carbs_g']:.1f} g")
        c4.metric("Fat", f"{nutrition['fat_g']:.1f} g")
        energy = pd.DataFrame({
            "kcal": [nutrition["protein_g"] * 4, nutrition["carbs_g"] * 4, nutrition["fat_g"] * 9]},
            index=["Protein", "Carbs", "Fat"])
        st.caption("Energy by macronutrient (kcal)")
        st.bar_chart(energy)
    else:
        st.info("No nutrition data for this class.")

st.caption("Estimates use typical per-100 g values and your portion size. Real meals vary; "
           "this is not medical or dietary advice.")
