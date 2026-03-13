import streamlit as st
import os

# 1. Verification Step
try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    import gdown
except ImportError as e:
    st.error(f"Dependency Error: {e}. Please ensure requirements.txt is correct.")
    st.stop()

st.set_page_config(page_title="Lung Classifier", page_icon="🫁")

# 2. Model Loading
@st.cache_resource
def load_model():
    file_id = '1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'lungs_disease_classifier.h5'
    
    if not os.path.exists(output):
        with st.spinner("Downloading model..."):
            gdown.download(url, output, quiet=False)
            
    return tf.keras.models.load_model(output)

model = load_model()
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia']

# 3. UI
st.title("🫁 Lung Disease Classifier")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button('Classify'):
        prediction = model.predict(img_array)
        idx = np.argmax(prediction[0])
        conf = np.max(prediction[0]) * 100
        
        st.success(f"Result: {CLASS_NAMES[idx]}")
        st.info(f"Confidence: {conf:.2f}%")
