import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_ID = "1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

st.title("Lung Disease Detection")

file = st.file_uploader("Upload X-ray Image", type=["jpg","jpeg","png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    if st.button("Predict"):
        prediction = model.predict(img)

        if prediction[0] > 0.5:
            st.error("Pneumonia Detected")
        else:
            st.success("Normal Lung")
