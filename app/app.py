import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Venomous Snake Identifier", page_icon="🐍")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'efficientnet_b0.keras')

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

st.title("🐍 Venomous vs Non-Venomous Snake Identifier")
st.write("Upload a photo of a snake and the model will predict whether it's venomous or non-venomous.")

st.warning("⚠️ This is an academic project for demonstration purposes only. Do not use this tool to make real-world safety decisions around snakes. Always treat unidentified snakes as potentially dangerous and consult local wildlife experts.")

uploaded_file = st.file_uploader("Choose a snake image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    # Note: EfficientNetB0 has built-in preprocessing, so no manual rescaling here

    with st.spinner('Analyzing...'):
        prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction")
    if prediction > 0.5:
        confidence = prediction * 100
        st.error(f"⚠️ Venomous ({confidence:.1f}% confidence)")
    else:
        confidence = (1 - prediction) * 100
        st.success(f"✅ Non-Venomous ({confidence:.1f}% confidence)")

    st.caption("Model: EfficientNetB0 (transfer learning) | Test accuracy: 87%")