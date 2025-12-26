import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load saved model directory using TFSMLayer for Keras 3 SavedModel format
# The endpoint name 'serve' was indicated during model.export()
model = tf.keras.models.Sequential([
    tf.keras.layers.TFSMLayer("saved_model", call_endpoint='serve')
])

IMG_SIZE = 224   # change if your model used a different size
CLASSES = ["Control Group", "DM Group"]  # update as per your labels

def predict_foot(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0

    if image.shape[-1] == 4:   # RGBA → RGB
        image = image[..., :3]

    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    # For binary classification with sigmoid, prediction is a single probability
    prediction_probability = preds[0][0]

    # Interpret the prediction
    if prediction_probability > 0.5:
        predicted_class_index = 1 # DM Group
    else:
        predicted_class_index = 0 # Control Group

    predicted_class_name = CLASSES[predicted_class_index]
    confidence = float(prediction_probability)

    return predicted_class_name, confidence

st.set_page_config(page_title="Diabetic Foot Risk Prediction", layout="centered")

st.title("🔥 Thermography-based Diabetic Foot Prediction")
st.write("Upload a plantar foot thermographic image")

uploaded_file = st.file_uploader(
    "Upload thermogram image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Thermogram", use_column_width=True)

    if st.button("Predict Risk"):
        with st.spinner("Analyzing thermogram..."):
            label, confidence = predict_foot(image)

        st.subheader("Prediction Result")
        st.success(f"Risk Category: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

st.info("""
This application is intended for research and educational purposes only.
It is NOT a diagnostic medical device.
""")
