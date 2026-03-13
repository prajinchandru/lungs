import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Lung Health Assistant", layout="wide")

st.title("🫁 Lung Disease Detection & Medical Assistance")

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=350)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    # Example prediction (replace with your real model)
    prediction = np.random.rand(3)

    index = prediction.argmax()
    disease = classes[index]

    st.subheader("🧠 AI Diagnosis")

    if disease == "Normal":

        st.success("No lung disease detected")

    else:

        st.error(f"Disease detected: {disease}")

        st.write("⚕ Please consult a pulmonologist.")

        st.subheader("📍 Nearby Hospitals")

        st.components.v1.html("""
        <script>
        navigator.geolocation.getCurrentPosition(function(position) {

            var lat = position.coords.latitude;
            var lon = position.coords.longitude;

            var iframe = document.createElement("iframe");

            iframe.width="100%";
            iframe.height="600";
            iframe.style.border="0";

            iframe.src="https://www.google.com/maps?q=hospitals&near="
            +lat+","+lon+"&output=embed";

            document.body.appendChild(iframe);

        });
        </script>
        """, height=600)
