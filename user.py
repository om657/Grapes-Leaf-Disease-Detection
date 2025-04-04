import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie

# Load the trained model
MODEL_PATH = "grape_disease_model.h5"  # Update this if needed
model = load_model(MODEL_PATH)

# Automatically get number of classes from the model
num_model_classes = model.output_shape[-1]  # Fetch model output size
print(f"Model predicts {num_model_classes} classes.")  # Debugging

# Define class labels dynamically based on model output size
POSSIBLE_CLASS_NAMES = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']  # Full list of possible classes
if num_model_classes > len(POSSIBLE_CLASS_NAMES):
    st.error("🚨 Model predicts more classes than expected. Update POSSIBLE_CLASS_NAMES!")
CLASS_NAMES = POSSIBLE_CLASS_NAMES[:num_model_classes]  # Trim list to match model output


# Function to preprocess the image (fixed shape issue)
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))  # Resize to match model input
    image = image.astype('float32') / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Function to predict disease
def predict_disease(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)[0]

    # Debugging: Print model output size
    num_classes = len(prediction)
    print(f"Model Output Size: {num_classes}")

    if num_classes != len(CLASS_NAMES):
        st.error(f"🚨 Mismatch! Model predicts {num_classes} classes, but CLASS_NAMES has {len(CLASS_NAMES)} entries.")
        return "Error", {}

    # Generate confidence scores dynamically
    confidence_scores = {CLASS_NAMES[i]: float(prediction[i]) for i in range(num_classes)}
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return predicted_class, confidence_scores


# Load Lottie animation safely
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# UI Layout
st.set_page_config(page_title="Grape Leaf Disease Detector", page_icon="🍇", layout="wide")

# Sidebar Info
st.sidebar.title("🔬 Model Information")
st.sidebar.info(f"""
- **Dataset:** Grape Leaf Disease Dataset  
- **Developer:** Om  
- **Domain:** Agriculture  
- **Detected Classes:** {', '.join(CLASS_NAMES)}
""")

# Main Title
st.title("🍇 Grape Leaf Disease Detection")

# Load and display Lottie animation (if available)
lottie_animation = load_lottiefile("animation.json")
if lottie_animation:
    st_lottie(lottie_animation, height=200, key="loading")
else:
    st.warning("⚠️ Animation file not found. Skipping animation.")

# File Uploader
uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        # Display uploaded image
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        # Predict
        predicted_class, confidence_scores = predict_disease(image)

        if predicted_class != "Error":
            # Show prediction result
            st.success(f"🌿 **Predicted Disease:** {predicted_class}")

            # Confidence score visualization
            st.subheader("📊 Confidence Scores:")
            st.bar_chart(confidence_scores)

            st.info("📌 Try uploading another image to test different cases!")
    else:
        st.error("🚨 Error loading image. Please upload a valid file.")

# Footer with clickable LinkedIn link
st.markdown('<br><hr><center>🚀 Developed by Om</center>', unsafe_allow_html=True)
st.markdown('<center><a href="https://www.linkedin.com/in/om07h" target="_blank">🔗 Connect on LinkedIn</a></center>',
            unsafe_allow_html=True)
