import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
from keras.models import load_model

st.set_page_config(page_title="Lung Disease Detection", page_icon="🫁")

FILE_ID = "1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

st.title("🫁 Lung Disease Detection AI")

file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

classes = ["Normal","Pneumonia","Other Lung Disease"]

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img = img.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    if st.button("Analyze"):
        pred = model.predict(img)
        index = pred.argmax()
        confidence = float(pred[0][index])

        st.success(f"Prediction: {classes[index]}")
        st.write(f"Confidence: {confidence:.2f}")
