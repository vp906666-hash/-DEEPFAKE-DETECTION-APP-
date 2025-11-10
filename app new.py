
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

# --- 1. Model Aur Setup ---
# Model file Streamlit app ke saath usi folder mein honi chahiye
MODEL_PATH = 'image_deepfake_detector_improved.h5' 
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model. Make sure '{MODEL_PATH}' is available.")
    st.stop()
    
# Haarcascade setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. Face Detection Function ---
def extract_face_from_image_app(image_bytes, target_size=(128, 128)):
    # Bytes ko NumPy array mein badalna
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None: return None 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_crop = frame[y:y+h, x:x+w]
        face_crop_resized = cv2.resize(face_crop, target_size)
        return face_crop_resized / 255.0
    return None

# --- 3. Streamlit Interface ---
st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🕵️ Deepfake Detection Project (VGG16)")
st.markdown("### Upload an Image to Analyze")

uploaded_file = st.file_uploader("Choose a JPG or PNG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        if st.button("Analyze and Predict"):
            with st.spinner('Analyzing Image for Deepfake features...'):
                uploaded_file.seek(0)
                face_data = extract_face_from_image_app(uploaded_file)

                if face_data is None:
                    st.error("🔴 ERROR: Face Not Detected in the image. Please upload a clear, frontal face image.")
                else:
                    input_array = np.expand_dims(face_data, axis=0)
                    prediction = model.predict(input_array)[0][0]
                    
                    # Overfitting Fix (Logic Ulatna) - Final Working Logic
                    NEW_THRESHOLD = 0.50 

                    st.markdown("---")
                    
                    if prediction < NEW_THRESHOLD: # Logic ulat diya gaya hai
                        st.balloons()
                        st.error(f"## 🚨 FAKE DETECTED!")
                        st.metric("Confidence Score", f"{(1-prediction)*100:.2f}% (Fake)")
                    else:
                        st.success(f"## ✅ REAL IMAGE")
                        st.metric("Confidence Score", f"{prediction*100:.2f}% (Real)")

