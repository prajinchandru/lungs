import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Lung Disease Classifier", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    # Ensure the model filename matches your uploaded file
    model = tf.keras.models.load_model('lungs_disease_classifier.h5')
    return model

model = load_model()

# Define the target image size based on model metadata
# The model requires input shape [null, 150, 150, 3]
IMG_SIZE = (150, 150)

# Define your class names (Update these to match your specific dataset)
# Since the model has 3 output units, there are 3 classes.
# Common examples: ["COVID-19", "Normal", "Pneumonia"]
CLASS_NAMES = ["Class 1", "Class 2", "Class 3"]

st.title("🫁 Lung Disease Classifier")
st.write("Upload a chest X-ray image to identify the condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis
    img_array = img_array / 255.0 # Normalize if your model was trained on scaled data

    # Prediction
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) # Use if model doesn't have internal softmax
            # Since your model already has a Softmax layer, we can use the raw prediction
            class_idx = np.argmax(predictions[0])
            confidence = 100 * np.max(predictions[0])

            st.success(f"Prediction: **{CLASS_NAMES[class_idx]}**")
            st.info(f"Confidence Level: {confidence:.2f}%")
