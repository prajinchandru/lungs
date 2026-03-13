import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# 1. Page Configuration
st.set_page_config(page_title="Lung Disease Classifier", page_icon="🫁")

# 2. Function to Download and Load Model
@st.cache_resource
def load_model_from_drive():
    # File ID extracted from your shared link
    file_id = '1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'lungs_disease_classifier.h5'
    
    # Download the file if it's not already there
    if not os.path.exists(output):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(url, output, quiet=False)
    
    # Load the model using Keras
    return tf.keras.models.load_model(output)

# Initialize model
try:
    model = load_model_from_drive()
except Exception as e:
    st.error(f"Failed to load model: {e}")

# 3. User Interface
st.title("🫁 Lung Disease Classifier")
st.write("Upload a chest X-ray image for analysis.")

# Note: Adjust these class names to match your model's specific training labels
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia']

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing to match the model's 150x150 input requirement
    img = image.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values

    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            predictions = model.predict(img_array)
            result_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            st.subheader(f"Prediction: {CLASS_NAMES[result_index]}")
            st.write(f"Confidence Level: {confidence:.2f}%")
