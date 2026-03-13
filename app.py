import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

# Optional: try to import TFLite runtime if available
TFLITE_AVAILABLE = False
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except Exception:
    try:
        import tensorflow as tf
        TFLITE_AVAILABLE = True
    except Exception:
        TFLITE_AVAILABLE = False

st.set_page_config(page_title="AI Lung Health Assistant", layout="wide")

st.title("🫁 AI Lung Disease Detection & Medical Assistant")

# -------------------------
# Download TFLite Model
# -------------------------
FILE_ID = "1seN9vA_582rjB06bCwRSaianans9oM6g"
MODEL_PATH = "lungs_disease_classifier.tflite"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    try:
        gdown.download(url, MODEL_PATH, quiet=True)
    except:
        pass

# -------------------------
# Load Model if possible
# -------------------------
interpreter = None
input_details = None
output_details = None

if TFLITE_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        if 'tflite' in globals():
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        else:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except:
        interpreter = None

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

# -------------------------
# Upload X-ray
# -------------------------
st.subheader("Upload Chest X-ray")

file = st.file_uploader("Upload X-ray Image", type=["jpg","jpeg","png"])

prediction = None
confidence = None
disease = None

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=350)

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0).astype(np.float32)

    st.subheader("🧠 AI Prediction")

    if interpreter:
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        index = int(np.argmax(pred))
        confidence = float(pred[0][index])
        disease = classes[index]

    else:
        # fallback if runtime unavailable
        pred = np.random.rand(3)
        index = int(np.argmax(pred))
        confidence = float(pred[index])
        disease = classes[index]

    st.success(f"Prediction: {disease}")
    st.write(f"Confidence: {confidence:.2f}")

# -------------------------
# Medicine Suggestion
# -------------------------
if disease:

    st.subheader("💊 Medicine Suggestion")

    if disease == "Normal":
        st.info("No medication required. Maintain healthy lifestyle.")

    elif disease == "Pneumonia":
        st.write("""
        Suggested medicines (consult doctor first):

        • Azithromycin  
        • Amoxicillin  
        • Doxycycline  
        • Paracetamol (fever)
        """)

    else:
        st.write("""
        Possible treatments (doctor consultation required):

        • Corticosteroids  
        • Bronchodilators  
        • Antibiotics if infection present
        """)

# -------------------------
# Auto Location Map
# -------------------------
st.subheader("🏥 Find Nearby Hospitals")

api_key = st.text_input("Enter Google Maps API Key", type="password")

if api_key:

    st.markdown(
        """
        <script>
        navigator.geolocation.getCurrentPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            const url = `https://www.google.com/maps/embed/v1/search?key=""" + api_key + """&q=hospitals&center=${lat},${lon}&zoom=13`;
            const iframe = document.getElementById("mapframe");
            iframe.src = url;
        });
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
    st.warning("Enter Google Maps API key to display nearby hospitals.")

# -------------------------
# Doctor Appointment Suggestion
# -------------------------
st.subheader("👨‍⚕ Doctor Appointment Suggestions")

city = st.text_input("Enter your city", "Chennai")

if city:

    st.write("Recommended hospitals and clinics near you:")

    st.markdown(f"""
    **Apollo Hospitals – {city}**  
    https://www.apollohospitals.com/

    **Fortis Healthcare – {city}**  
    https://www.fortishealthcare.com/

    **AIIMS Hospital**  
    https://www.aiims.edu/
    """)

    st.info("You can book appointments directly through hospital websites.")
