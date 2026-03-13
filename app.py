import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------
# PAGE SETTINGS
# -------------------------
st.set_page_config(
    page_title="Lung Disease AI",
    page_icon="🫁",
    layout="centered"
)

# -------------------------
# MODEL DOWNLOAD
# -------------------------
FILE_ID = "1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------
# UI HEADER
# -------------------------
st.title("🫁 Lung Disease Detection AI")
st.write("Upload a Chest X-ray image to analyze lung condition.")

# -------------------------
# FILE UPLOAD
# -------------------------
file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# CLASS NAMES
# -------------------------
classes = [
    "Normal",
    "Pneumonia",
    "Other Lung Disease"
]

# -------------------------
# PREDICTION
# -------------------------
if file is not None:

    image = Image.open(file).convert("RGB")

    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    if st.button("Analyze X-ray"):

        prediction = model.predict(img)

        class_index = np.argmax(prediction)

        confidence = float(prediction[0][class_index])

        st.subheader("Prediction Result")

        st.success(
            f"Prediction: {classes[class_index]}"
        )

        st.write(
            f"Confidence: {confidence:.2f}"
        )

st.markdown("---")
st.caption("AI Medical Assistant – Lung X-ray Analysis")
