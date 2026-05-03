import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="AgroAI Ethio", layout="centered")

st.title("🍅 AgroAI Ethio - Real Tomato Leaf Disease Detection")
st.write("Upload a tomato leaf image for AI diagnosis")

# ✅ LOAD REAL MODEL
model = tf.keras.models.load_model("plant_full_model.h5")

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

# ✅ PREPROCESS IMAGE
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

file = st.file_uploader("Upload Tomato Leaf Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Leaf")

    # ✅ REAL AI PREDICTION
    x = preprocess(img)
    pred = model.predict(x)

    index = np.argmax(pred)
    confidence = np.max(pred)

    st.subheader("🔍 AI Result")
    st.success(f"Disease: {labels[index]}")
    st.info(f"Confidence: {round(confidence * 100, 2)}%")

    if labels[index] == "Healthy":
        st.success("Plant is healthy 🌿")
    else:
        st.warning("Disease detected ⚠️ Take action immediately")

st.caption("AgroAI Ethio - by Ali Umar 2026")