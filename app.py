import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("saveModel.keras")

# Define class names (optional)
class_names = ['Fake', 'Real']

# Prediction function
def predict_image_class(uploaded_image):
    img = uploaded_image.resize((224, 224))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    confidence = float(predictions[0][0])
    predicted_class = int(confidence > 0.2)  # Using 0.2 as threshold

    return class_names[predicted_class], confidence

# Streamlit UI
st.title("Deepfake Image Detection")
st.write("Upload an image to check if it's real or fake using a CNN model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_container_width=True)

    # Predict
    if st.button("Detect"):
        label, conf = predict_image_class(image_display)
        st.success(f"Prediction: **{label}**")
        # st.info(f"Confidence Score: **{conf:.2f}**")
