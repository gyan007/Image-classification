import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TensorFlow model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/imageClassfierMode.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Predict function
def predict_image(model, image):
    image = image.resize((256, 256))  # Resize to fit the model input
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Prediction
    prediction = model.predict(img_array)
    return 'Person is Happy' if prediction[0] < 0.5 else 'Person is Sad'

# Streamlit app layout
st.title("Images classification for happy or sad face")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model
    model = load_model()

    # Predict button
    if st.button('Predict'):
        if model:
            result = predict_image(model, image)
            st.write(f"Prediction: {result}")
