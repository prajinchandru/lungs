import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Lung Detection AI", layout="centered")

# -----------------------------
# Download model from Google Drive
# -----------------------------

file_id = "1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY"
model_path = "model.h5"

if not os.path.exists(model_path):
    st.info("Downloading AI model from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# -----------------------------
# Load AI Model
# -----------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🫁 Lung Disease Detection AI")
st.write("Upload a Chest X-Ray image to analyze lung condition.")

uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "png", "jpeg"]
)

# -----------------------------
# Image Processing
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    if st.button("Predict"):

        prediction = model.predict(img)

        confidence = float(prediction[0])

        st.subheader("Prediction Result")

        if confidence > 0.5:
            st.error(f"Pneumonia Detected ⚠️ (Confidence: {confidence:.2f})")
        else:
            st.success(f"Normal Lung ✅ (Confidence: {1-confidence:.2f})")

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.write("AI Powered Lung X-ray Detection")
