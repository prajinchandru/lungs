import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Lung Disease AI Assistant", layout="wide")

st.title("🫁 Lung Disease Detection & Medical Assistant")

st.write("Upload a Chest X-ray image to analyze lung condition and get medical help.")

# -----------------------------
# File Upload
# -----------------------------

file = st.file_uploader("Upload Chest X-ray", type=["jpg","jpeg","png"])

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=300)

    # Dummy prediction (replace with real model later)
    prediction = np.random.rand(3)

    index = prediction.argmax()
    confidence = float(prediction[index])

    disease = classes[index]

    st.subheader("🧠 AI Prediction")

    st.success(f"Prediction: {disease}")
    st.write(f"Confidence: {confidence:.2f}")

    # -----------------------------
    # Medicine Suggestion
    # -----------------------------

    st.subheader("💊 Medicine Suggestion")

    if disease == "Normal":
        st.info("No medicine needed. Maintain healthy lifestyle.")

    elif disease == "Pneumonia":
        st.write("""
        Suggested medicines (consult doctor before use):

        • Azithromycin  
        • Amoxicillin  
        • Doxycycline  
        • Paracetamol for fever
        """)

    else:
        st.write("""
        Suggested medicines (consult doctor):

        • Corticosteroids  
        • Bronchodilators  
        • Antibiotics if infection present
        """)

# -----------------------------
# Google Maps Section
# -----------------------------

st.subheader("🏥 Find Nearby Hospitals")

api_key = st.text_input("Enter Google Maps API Key", type="password")

if api_key:

    location = st.text_input("Enter your city", "Chennai")
    zoom = 12

    map_url = f"https://www.google.com/maps/embed/v1/search?key={api_key}&q=hospitals+near+{location}&zoom={zoom}"

    st.components.v1.iframe(map_url, height=600)

else:
    st.warning("Enter Google Maps API key to show hospitals.")
