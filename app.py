import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Lung Health Assistant", layout="wide")

st.title("🫁 Lung Disease Detection System")

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=350)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    # Replace with your real model prediction
    prediction = np.random.rand(3)

    index = prediction.argmax()
    disease = classes[index]

    st.subheader("🧠 AI Diagnosis")

    if disease == "Normal":

        st.success("No lung disease detected")

    else:

        st.error(f"Disease detected: {disease}")

        st.write("⚕ Please consult a pulmonologist immediately.")

        st.subheader("📍 Finding Hospitals Near You")

        st.info("Click the button to detect your live location and open nearby hospitals.")

        if st.button("Find Hospitals Near Me"):

            st.markdown("""
            <script>
            navigator.geolocation.getCurrentPosition(function(position) {

                var lat = position.coords.latitude;
                var lon = position.coords.longitude;

                var maps_url =
                "https://www.google.com/maps/search/hospitals/@"
                + lat + "," + lon + ",14z";

                window.open(maps_url);

            });
            </script>
            """, unsafe_allow_html=True)
