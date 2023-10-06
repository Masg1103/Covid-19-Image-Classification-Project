import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Create the title for the app
st.title("COVID-19 Lung X-ray Image Classification")
st.write("Upload Lung X-ray images for COVID-19 prediction")

# Load the trained CNN COVID-19 prediction model model
model = load_model("covid_cnn.hdf5")

# Define class labels
class_labels = ["Covid", "Normal", "Viral Pneumonia"]

# Function to preprocess and predict an image
def predict_image(image):
    # Convert the stream to a PIL Image
    pil_image = Image.open(image)
    pil_image = Image.convert("RGB")

     # Preprocess the image
    pil_image = pil_image.resize((128, 128))  # Resize to the input size expected by the model
    image = np.array(pil_image)  # Convert to NumPy array
    image = image / 255.0  # Normalize pixel values (assuming model expects values in [0, 1])

    # Make a prediction
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = prediction[np.argmax(prediction)]

    return predicted_class, confidence

# Allow users to upload images
uploaded_images = st.file_uploader("Upload X-ray images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make a prediction
        predicted_class, confidence = predict_image(uploaded_image)

        # Display the prediction
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")