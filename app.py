import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf
from plant_info import plant_info

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Medicinal Plant Identification",
    layout="centered"
)

# ------------------ CACHE MODEL ------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("medicinal_plant_mobilenetv2_final.keras")

model = load_model()

# ------------------ CACHE CLASS NAMES ------------------
@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

class_names = load_class_names()

# ------------------ CONSTANTS ------------------
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.90  # 90%

# ------------------ UI ------------------
st.title("🌿 Medicinal Plant Identification System")
st.write("Upload a plant image or capture one using the camera.")
st.info("📌 Best results when you upload a close-up image of the leaf (not the full plant or flowers).")


st.subheader("📸 Choose Image Source")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

camera_file = st.camera_input("Take a photo")

# ------------------ PREDICTION ------------------
if uploaded_file is not None or camera_file is not None:

    # Load image
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)

    confidence = float(np.max(predictions))
    predicted_index = int(np.argmax(predictions))
    plant_name = class_names[predicted_index]

    # ------------------ CONFIDENCE REJECTION ------------------
    if confidence < CONFIDENCE_THRESHOLD:
        st.error("❌ Unable to confidently identify this plant")
        st.write("Please upload a clearer leaf image or try another angle.")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
        st.stop()

    # ------------------ RESULT ------------------
    st.success(f"🌱 Predicted Plant: **{plant_name}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")

    st.subheader("💊 Medicinal Usage")
    st.write(
        plant_info.get(
            plant_name,
            "Medicinal usage information not available."
        )
    )

# ------------------ DISCLAIMER ------------------
st.markdown("---")
st.warning(
    "⚠️ This system is for educational purposes only. "
    "Consult a qualified expert before using plants for medicinal purposes."
)
