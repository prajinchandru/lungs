import streamlit as st
import os

# Set page config first
st.set_page_config(page_title="Lung Disease Classifier", page_icon="🫁")

# Try-except block to help debug installation issues in the UI
try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    import gdown
except ImportError as e:
    st.error(f"Installation Error: {e}. Check your requirements.txt")
    st.stop()

# 1. Download/Load Model
@st.cache_resource
def load_model():
    file_id = '1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'lungs_disease_classifier.h5'
    
    if not os.path.exists(output):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, output, quiet=False)
            
    return tf.keras.models.load_model(output)

model = load_model()

# 2. Define Classes (Match your model's training labels)
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia']

# 3. UI logic
st.title("🫁 Lung Disease Classifier")
st.write("Upload a chest X-ray image for instant classification.")

uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_container_width=True)
    
    # Preprocessing to 150x150
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            prediction = model.predict(img_array)
            result = CLASS_NAMES[np.argmax(prediction[0])]
            confidence = np.max(prediction[0]) * 100
            
            st.success(f"Prediction: {result}")
            st.info(f"Confidence: {confidence:.2f}%")
