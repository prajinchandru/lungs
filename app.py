import streamlit as st
import os

# Set page config
st.set_page_config(page_title="Lung Disease Classifier", page_icon="🫁")

# Defensive imports to catch errors in the UI
try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    import gdown
except ImportError as e:
    st.error(f"Module loading failed: {e}. Please wait for the installer to finish.")
    st.stop()

# --- Model Loading ---
@st.cache_resource
def load_classifier():
    model_path = 'lungs_disease_classifier.h5'
    file_id = '1yL5KwOn8RTHq5ANhK6qSf_93ccxyGQeY'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # If the file isn't in the GitHub repo, download it
    if not os.path.exists(model_path):
        with st.spinner("Fetching model..."):
            gdown.download(url, model_path, quiet=False)
            
    return tf.keras.models.load_model(model_path)

model = load_classifier()
CLASS_NAMES = ['COVID-19', 'Normal', 'Pneumonia']

# --- UI ---
st.title("🫁 Lung Disease Classification")
st.info("Upload a chest X-ray image for an automated AI diagnostic prediction.")

file = st.file_uploader("Upload X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert('RGB')
    st.image(img, caption="Target Image", use_container_width=True)
    
    # Match your model's 150x150 input requirement
    resized_img = img.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    if st.button("Run Prediction"):
        prediction = model.predict(img_array)
        top_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        
        st.subheader(f"Result: {CLASS_NAMES[top_idx]}")
        st.write(f"Confidence Level: {confidence:.2f}%")
