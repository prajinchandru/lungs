import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os
import tflite_runtime.interpreter as tflite

FILE_ID = "YOUR_DRIVE_ID"
MODEL_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Lung Disease Detection")

file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])

    if pred[0] > 0.5:
        st.error("Pneumonia Detected")
    else:
        st.success("Normal Lung")
