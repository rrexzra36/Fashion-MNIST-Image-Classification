import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations to avoid compatibility issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Show only errors (not warnings/info)

import streamlit as st
import tensorflow as tf

import numpy as np
from PIL import Image, ImageOps
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="centered"
)

# Function to load the model (with cache for faster loading)
@st.cache_resource
def load_model():
    """Load the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model('model/cnn_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

# List of class names according to the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- Streamlit Interface ---
st.title("👕 Fashion MNIST Clothing Classifier")
st.write(
    "Upload an image of a clothing item (such as a t-shirt, pants, shoes) "
    "and the model will try to predict it. Make sure the image has a white/plain background for more accurate predictions."
)

# File upload option
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None and model is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert('L') # Convert to grayscale

    with col1:
        st.image(image, caption='Uploaded Image', width=200)

    # Preprocess the image to fit the model input
    image_resized = image.resize((28, 28))
    image_inverted = ImageOps.invert(image_resized)
    image_array = np.array(image_inverted) / 255.0
    image_final = np.expand_dims(image_array, axis=[0, -1])

    with col2:
        st.image(image_final.squeeze(), caption='Preprocessed Image (28x28, Inverted)', width=200, channels="GRAY")

    if st.button('Classify This Image!'):
        with st.spinner('Model is thinking... 🤔'):
            prediction = model.predict(image_final)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class_index]
            confidence = np.max(prediction) * 100

            st.success(f"**Model Prediction:** {predicted_class_name}")
            st.info(f"**Confidence Level:** {confidence:.2f}%")

            st.write("Probability for each class:")
            prob_df = pd.DataFrame({
                'Class': class_names,
                'Probability': prediction[0]
            })
            prob_df = prob_df.set_index('Class')

            fig, ax = plt.subplots()
            prob_df.plot(kind='bar', legend=False, ax=ax)
            ax.set_ylabel('Probability')
            ax.set_xlabel('Class')
            plt.xticks(rotation=45)
            st.pyplot(fig)
else:
    if model is None:
        st.warning("Model could not be loaded. Make sure the file 'fashion_mnist_model.h5' exists in the same directory.")

st.sidebar.header("About Project")
st.sidebar.info(
    "This app uses a **Convolutional Neural Network (CNN)** model "
    "trained on the **Fashion MNIST** dataset to classify 10 types of clothing items. "
    "Built with TensorFlow/Keras and Streamlit."
)