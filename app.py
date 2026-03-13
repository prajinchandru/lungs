import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(page_title="Lung Disease Detection AI", page_icon="🫁")

# Google Drive model
FILE_ID = "1seN9vA_582rjB06bCwRSaianans9oM6g"
MODEL_PATH = "lungs_disease_classifier.tflite"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

st.title("🫁 Lung Disease Detection AI")
st.write("Upload a chest X-ray image")

file = st.file_uploader("Upload X-ray", type=["jpg","jpeg","png"])

classes = [
    "Normal",
    "Pneumonia",
    "Other Lung Disease"
]

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray")

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    if st.button("Analyze"):

        # Dummy prediction simulation
        # (Replace with real model inference later)

        prediction = np.random.rand(3)
        index = prediction.argmax()
        confidence = float(prediction[index])

        st.success(f"Prediction: {classes[index]}")
        st.write(f"Confidence: {confidence:.2f}")
