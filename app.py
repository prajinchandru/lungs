import streamlit as st
import os

# Set page config
st.set_page_config(page_title="Lung Disease Classifier", page_icon="🫁")

# Check for dependencies
try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    import gdown
except ImportError as e:
    st.error(f"Dependencies are still installing or failed: {e}")
    st.stop()

# 1. Load Model
@st.cache_resource
def load_model():
    model_path = 'lungs_disease_classifier.h5'
    
    # If the file isn't in the repo for some reason, download it
    if not os.path.exists(model_path):
        file_id = '1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY'
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.spinner("Downloading model..."):
            gdown.download(url, model_path, quiet=False)
            
    return tf.keras.models.load_model(model_path)

model = load_model()

# Labels (Ensure these match your training order)
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia']

# 2. UI
st.title("🫁 Lung Disease Classifier")
st.write("Upload a chest X-ray image for classification.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing (Model input: 150x150)
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array)
            result = CLASS_NAMES[np.argmax(prediction[0])]
            confidence = np.max(prediction[0]) * 100
            
            st.success(f"Result: {result}")
            st.info(f"Confidence: {confidence:.2f}%")
