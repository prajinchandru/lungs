import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Lung Health Assistant", layout="wide")

st.title("🫁 Lung Disease Detection & Medical Assistant")

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=350)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    # Example prediction (replace with real model)
    prediction = np.random.rand(3)

    index = prediction.argmax()
    disease = classes[index]

    st.subheader("🧠 AI Result")

    if disease == "Normal":

        st.success("No lung disease detected ✅")

    else:

        st.error(f"Disease detected: {disease}")

        st.subheader("⚕ Recommended Medical Help")

        st.write("Consult a **Pulmonologist (lung specialist)** immediately.")

        st.subheader("📍 Find Nearby Hospitals")

        st.markdown("""
        <script>
        navigator.geolocation.getCurrentPosition(function(position) {

            var lat = position.coords.latitude;
            var lon = position.coords.longitude;

            var map_url = "https://www.google.com/maps?q=hospitals&near=" + lat + "," + lon;

            window.open(map_url);

        });
        </script>
        """, unsafe_allow_html=True)

        st.info("Click the button below to find nearby hospitals.")

        if st.button("Find Hospitals Near Me"):

            st.markdown("""
            <script>
            navigator.geolocation.getCurrentPosition(function(position) {

                var lat = position.coords.latitude;
                var lon = position.coords.longitude;

                var map_url = "https://www.google.com/maps?q=hospitals&near=" + lat + "," + lon;

                window.open(map_url);

            });
            </script>
            """, unsafe_allow_html=True)

        st.subheader("👨‍⚕ Suggested Doctors")

        st.write("""
        **Pulmonology Specialists**

        • Apollo Hospitals  
        https://www.apollohospitals.com/

        • Fortis Healthcare  
        https://www.fortishealthcare.com/

        • AIIMS Hospital  
        https://www.aiims.edu/
        """)
