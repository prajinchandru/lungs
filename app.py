import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="Lung Disease Detection AI", page_icon="🫁")

FILE_ID = "1seN9vA_582rjB06bCwRSaianans9oM6g"
MODEL_PATH = "lungs_disease_classifier.tflite"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ["Normal", "Pneumonia", "Other Lung Disease"]

st.title("🫁 Lung Disease Detection AI")
st.write("Upload a chest X-ray image")

file = st.file_uploader("Upload X-ray", type=["jpg","jpeg","png"])

if file:

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded X-ray")

    img = image.resize((150,150))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0).astype(np.float32)

    if st.button("Analyze"):

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        pred = interpreter.get_tensor(output_details[0]['index'])

        index = pred.argmax()
        confidence = float(pred[0][index])

        st.success(f"Prediction: {classes[index]}")
        st.write(f"Confidence: {confidence:.2f}")
