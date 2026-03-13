import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Lung Health Assistant", layout="wide")

st.title("🫁 Lung Disease Detection & Medical Assistant")

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

# Upload X-ray
file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=350)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    # Example prediction (replace with your model)
    prediction = np.random.rand(3)

    index = prediction.argmax()
    disease = classes[index]

    st.subheader("🧠 AI Result")

    if disease == "Normal":

        st.success("No lung disease detected ✅")

    else:

        st.error(f"Disease detected: {disease}")

        st.subheader("⚕ Medical Help Recommendation")

        st.write("""
        We recommend consulting a **lung specialist (Pulmonologist)** immediately.
        """)

        # -----------------------------
        # Get live location
        # -----------------------------

        st.markdown("### 📍 Find Nearby Hospitals")

        api_key = st.text_input("Enter Google Maps API Key", type="password")

        if api_key:

            st.markdown(
            f"""
            <script>
            navigator.geolocation.getCurrentPosition(
            (position) => {{
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;

                const map = document.getElementById("mapframe");

                map.src =
                "https://www.google.com/maps/embed/v1/search?key={api_key}&q=hospitals&center=" + lat + "," + lon + "&zoom=13";
            }});
            </script>
            """,
            unsafe_allow_html=True
            )

            st.components.v1.html(
            """
            <iframe
            id="mapframe"
            width="100%"
            height="600"
            style="border:0"
            loading="lazy">
            </iframe>
            """,
            height=600
            )

        else:

            st.warning("Enter Google Maps API key to show hospitals.")

        # -----------------------------
        # Doctor suggestions
        # -----------------------------

        st.subheader("👨‍⚕ Suggested Doctors")

        st.write("""
        **Pulmonologist Specialists**

        • Apollo Hospitals  
        https://www.apollohospitals.com/

        • Fortis Healthcare  
        https://www.fortishealthcare.com/

        • AIIMS Hospital  
        https://www.aiims.edu/
        """)
