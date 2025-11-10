import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os
import requests 

# --- 1. Model Download Aur Setup ---
MODEL_PATH = 'model.h5' 
FILE_ID = "1AHj92kN9KG1O1U1vvcp6-iqRJMMIwPEY" # Aapki File ID

@st.cache_resource
def download_model(file_id):
    st.write("Downloading model from Google Drive...")
    # Direct Google Drive Download URL (Sabse Stable Format)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    response = requests.get(download_url, stream=True)
    
    # Download karna
    with open(MODEL_PATH, 'wb') as file:
        for data in response.iter_content(chunk_size=1024*1024): # 1MB chunks
            file.write(data)
    
    return load_model(MODEL_PATH)

try:
    model = download_model(FILE_ID) 
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"🔴 ERROR: Model download/load nahi ho paya. Check Drive Link/File ID. Error: {e}")
    st.stop()
    
# Haarcascade setup
HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(HAARCASCADE_PATH):
    # Yeh part Streamlit Cloud par aam taur par fail nahi hota
    st.error("Haarcascade file not found!")
    st.stop()
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

# --- 2. Face Detection Function ---
def extract_face_from_image_app(image_bytes, target_size=(128, 128)):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None: return None, None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    
    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_crop = frame[y:y+h, x:x+w]
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        resized_face = cv2.resize(face_crop_rgb, target_size)
        normalized_face = resized_face / 255.0
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return normalized_face, frame
    else:
        return None, frame

# --- 3. Streamlit UI ---
st.title("👁️ Deepfake Image Detector")
st.markdown("Upload an image to determine if it is Real or a Synthetic Deepfake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    if st.button("Analyze Image"):
        st.subheader("Analysis Results:")
        
        uploaded_file.seek(0)
        
        face_input, frame_with_box = extract_face_from_image_app(uploaded_file)
        
        if face_input is None:
            st.warning("⚠️ No clear face detected in the image.")
            st.image(frame_with_box, caption="Original Image (No face detected)", use_column_width=True, channels="BGR")
        else:
            st.image(frame_with_box, caption="Detected Face", use_column_width=True, channels="BGR")
            
            # Prediction
            face_input_expanded = np.expand_dims(face_input, axis=0)
            prediction = model.predict(face_input_expanded)[0][0]
            
            # --- CONFIDENCE FIX: Yahan confidence aur label define ho rahe hain ---
            confidence = prediction if prediction > 0.5 else 1 - prediction
            label = "DEEPFAKE" if prediction > 0.5 else "REAL"
            # ---------------------------------------------------------------------

            st.markdown("---")
            st.metric(label="Predicted Label", value=label)
            st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
            
            if label == "DEEPFAKE":
                st.error("🚨 Warning: This image is likely a **Deepfake**.")
            else:
                st.success("✅ Prediction: This image appears to be **Real**.")
