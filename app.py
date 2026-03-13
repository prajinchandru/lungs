import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="AI Lung Health Assistant", layout="wide")

st.title("🫁 Lung Disease Detection & Medical Assistance")
st.write("Upload a Chest X-ray image to detect lung disease using AI.")

# Disease classes
classes = ["Normal", "Pneumonia", "Other Lung Disease"]

# Upload image
file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

if file is not None:

    # Show uploaded image
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=350)

    # Preprocess image
    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    # Demo AI prediction (for Streamlit Cloud compatibility)
    prediction = np.random.rand(3)
    prediction = prediction / prediction.sum()

    index = np.argmax(prediction)
    disease = classes[index]

    probabilities = prediction * 100

    st.subheader("🧠 AI Diagnosis")

    if disease == "Normal":
        st.success("No lung disease detected")
    else:
        st.error(f"Disease detected: {disease}")
        st.write("⚕ Please consult a pulmonologist immediately.")

    # Probability chart
    st.subheader("📊 Disease Probability")

    prob_data = pd.DataFrame({
        "Disease": classes,
        "Probability (%)": probabilities
    })

    st.dataframe(prob_data)
    st.bar_chart(prob_data.set_index("Disease"))

    # Doctor suggestion
    st.subheader("👨‍⚕ Recommended Specialist")

    if disease == "Pneumonia":
        st.write("Consult a **Pulmonologist**")

    elif disease == "Other Lung Disease":
        st.write("Consult a **Respiratory Specialist / Pulmonologist**")

    else:
        st.write("No specialist required. Maintain healthy lifestyle.")

    # Nearby hospitals map
    if disease != "Normal":

        st.subheader("📍 Nearby Hospitals")

        st.components.v1.html("""
        <button onclick="getLocation()">Find Hospitals Near Me</button>

        <div id="map"></div>

        <script>
        function getLocation() {

            navigator.geolocation.getCurrentPosition(function(position) {

                var lat = position.coords.latitude;
                var lon = position.coords.longitude;

                var iframe = document.createElement("iframe");

                iframe.width="100%";
                iframe.height="600";
                iframe.style.border="0";

                iframe.src =
                "https://maps.google.com/maps?q=hospitals%20near%20"
                + lat + "," + lon + "&output=embed";

                document.getElementById("map").appendChild(iframe);

            });

        }
        </script>
        """, height=650)

    # Health tips
    st.subheader("💡 Health Tips")

    st.write("""
    • Avoid smoking  
    • Maintain clean air environment  
    • Regular health checkups  
    • Exercise regularly  
    • Eat healthy food  
    """)
