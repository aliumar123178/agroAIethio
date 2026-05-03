import streamlit as st
import numpy as np
from PIL import Image
import random

st.set_page_config(page_title="AgroAI Ethio", layout="centered")

st.title("🍅 AgroAI Ethio - Tomato Leaf Disease Detection (Demo)")
st.write("Upload a tomato leaf image")

labels = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl Virus",
    "Mosaic Virus",
    "Healthy"
]


def  real():
    index = random.randint(0, len(labels)-1)
    confidence = random.uniform(0.70, 0.99)
    return index, confidence

file = st.file_uploader("Upload Tomato Leaf Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Leaf")

    index, confidence = fake_predict()

    st.subheader("🔍 AI Result (Demo Mode)")
    st.success(f"Disease: {labels[index]}")
    st.info(f"Confidence: {round(confidence*100,2)}%")

    if labels[index] == "Healthy":
        st.success("Plant is healthy 🌿")
    else:
        st.warning("Disease detected ⚠️ (Demo result)")

st.caption("AgroAI Ethio -  by Ali Umar")