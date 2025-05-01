import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
import io
import hashlib
import random
import os
import keras
import numpy as np
from keras import applications
import requests
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax" 

# ----- Webcam Capture Transformer -----
class CaptureImageTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture = False
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.capture:
            self.frame = img
            self.capture = False
        return img

# ----- Image Fetching Functions -----
def fetch_real_image():
    real_dir = "game_real"
    if not os.path.exists(real_dir):
        st.error("Missing 'game_real' directory")
        return None
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not real_images:
        st.error("No real images found")
        return None
    if "used_real_images" not in st.session_state:
        st.session_state.used_real_images = set()
    available_images = [img for img in real_images if img not in st.session_state.used_real_images]
    if not available_images:
        st.session_state.used_real_images = set()
        available_images = real_images
    selected_image = random.choice(available_images)
    st.session_state.used_real_images.add(selected_image)
    try:
        return Image.open(selected_image).copy()
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

def fetch_fake_image():
    fake_dir = "Game_Fake"
    if not os.path.exists(fake_dir):
        st.error("Missing 'Game_Fake' directory")
        return None
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not fake_images:
        st.error("No fake images found")
        return None
    selected_image = random.choice(fake_images)
    try:
        return Image.open(selected_image).copy()
    except Exception as e:
        st.error(f"Error loading fake image: {str(e)}")
        return None

# ----- AI Detection Functions -----
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model("deepfake_detection_model.h5", compile=False)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def get_image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()

@st.cache_data(show_spinner=False)
def predict_image(image_hash: str, _image: Image.Image):
    model = load_model()
    if model is None:
        return [{"label": "error", "score": 1.0}]
    img = _image.convert('RGB').resize((224, 224))
    img_array = applications.efficientnet.preprocess_input(np.array(img))
    try:
        prob = model.predict(np.expand_dims(img_array, axis=0))[0][0]
        return [
            {"label": "real", "score": float(prob)},
            {"label": "fake", "score": float(1 - prob)}
        ] if prob > 0.5 else [
            {"label": "fake", "score": float(1 - prob)},
            {"label": "real", "score": float(prob)}
        ]
    except Exception as e:
        return [{"label": "error", "score": 1.0}]

# ----- Sightengine API Integration -----
def analyze_with_sightengine(image_bytes, api_user, api_secret):
    try:
        response = requests.post(
            'https://api.sightengine.com/1.0/check.json',
            files={'media': ('image.jpg', image_bytes, 'image/jpeg')},
            data={
                'api_user': api_user,
                'api_secret': api_secret,
                'models': 'deepfake,genai'
            }
        )
        result = response.json()
        scores = {
            'deepfake': result['type'].get('deepfake', 0.0),
            'ai_generated': result['type'].get('ai_generated', 0.0)
        }
        return scores
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# ----- Streamlit UI -----
st.set_page_config(page_title="DeepShield", page_icon="🕵️", layout="centered")
st.title("DeepShield - Deepfake Detection")

st.markdown("### 📤 Upload or Capture Image")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
image = None

st.markdown("### 🎥 Capture from Webcam")
ctx = webrtc_streamer(
    key="camera",
    video_transformer_factory=CaptureImageTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_transformer and st.button("📸 Capture & Analyze"):
    ctx.video_transformer.capture = True
    st.info("Capturing image...")
    while ctx.video_transformer.frame is None:
        pass
    frame_np = ctx.video_transformer.frame
    image = Image.fromarray(frame_np[..., ::-1])  # BGR to RGB
    st.image(image, caption="Captured Image", use_column_width=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

if image is not None:
    image_hash = get_image_hash(image)
    results = predict_image(image_hash, image)
    scores = {r["label"]: r["score"] for r in results}

    st.markdown("### 🧠 Prediction Results")
    st.markdown(f"**Real Confidence:** {scores['real']*100:.2f}%")
    st.markdown(f"**Fake Confidence:** {scores['fake']*100:.2f}%")
