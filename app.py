import streamlit as st
import os

# 1. Setup & Error Handling for Imports
try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    import gdown
except ImportError as e:
    st.error(f"Dependencies are still installing or failed: {e}")
    st.stop()

st.set_page_config(page_title="Lung Disease Classifier", page_icon="🫁")

# 2. Model Loading Logic
@st.cache_resource
def load_model():
    model_file = 'lungs_disease_classifier.h5'
    # Fallback to Google Drive if file is missing in Repo
    if not os.path.exists(model_file):
        file_id = '1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY'
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.spinner("Downloading model from Drive..."):
            gdown.download(url, model_file, quiet=False)
            
    return tf.keras.models.load_model(model_file)

model = load_model()
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia']

# 3. UI
st.title("🫁 Lung Disease Classifier")
st.write("Upload a chest X-ray for AI analysis.")

uploaded_file = st.file_uploader("Select Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Scan", use_container_width=True)
    
    # Preprocessing
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button("Classify"):
        with st.spinner("Processing..."):
            preds = model.predict(img_array)
            idx = np.argmax(preds[0])
            conf = np.max(preds[0]) * 100
            
            st.success(f"Result: {CLASS_NAMES[idx]}")
            st.info(f"Confidence: {conf:.2f}%")
