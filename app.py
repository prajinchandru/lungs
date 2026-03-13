import streamlit as st
import numpy as np
from PIL import Image
import requests
import base64

st.set_page_config(page_title="Lung Disease Detection AI", page_icon="🫁")

st.title("🫁 Lung Disease Detection AI")
st.write("Upload a chest X-ray image")

file = st.file_uploader("Upload X-ray", type=["jpg","jpeg","png"])

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray")

    img = image.resize((150,150))
    img = np.array(img)/255.0

    if st.button("Analyze"):

        # encode image
        img_bytes = base64.b64encode(img.tobytes()).decode()

        response = requests.post(
            "https://lungs-ai-api.onrender.com/predict",
            json={"image": img_bytes}
        )

        result = response.json()

        prediction = result["class"]
        confidence = result["confidence"]

        st.success(f"Prediction: {classes[prediction]}")
        st.write(f"Confidence: {confidence:.2f}")
