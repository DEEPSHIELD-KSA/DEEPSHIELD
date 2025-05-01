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

# Set Keras backend
os.environ["KERAS_BACKEND"] = "jax"

# ----- Webcam Capture Transformer -----
class CaptureImageTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.capture:
            st.session_state.captured_frame = img
            self.capture = False
        return img

# ----- Image Handling Functions -----
def fetch_real_image():
    real_dir = "game_real"
    if not os.path.exists(real_dir):
        return None
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not real_images:
        return None
    if "used_real_images" not in st.session_state:
        st.session_state.used_real_images = set()
    available = [img for img in real_images if img not in st.session_state.used_real_images]
    if not available:
        st.session_state.used_real_images = set()
        available = real_images
    selected = random.choice(available)
    st.session_state.used_real_images.add(selected)
    try:
        return Image.open(selected)
    except:
        return None

def fetch_fake_image():
    fake_dir = "Game_Fake"
    if not os.path.exists(fake_dir):
        return None
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not fake_images:
        return None
    selected = random.choice(fake_images)
    try:
        return Image.open(selected)
    except:
        return None

# ----- AI Model Functions -----
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
        return []
    img = _image.convert('RGB').resize((224, 224))
    img_array = applications.efficientnet.preprocess_input(np.array(img))
    try:
        prob = model.predict(np.expand_dims(img_array, axis=0))[0][0]
        return [
            {"label": "real", "score": float(prob)},
            {"label": "fake", "score": float(1 - prob)}
        ]
    except:
        return []

# ----- Sightengine API Integration -----
def analyze_with_sightengine(image):
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        response = requests.post(
            'https://api.sightengine.com/1.0/check.json',
            files={'media': buf.getvalue()},
            data={
                'api_user': st.secrets["sightengine_user"],
                'api_secret': st.secrets["sightengine_secret"],
                'models': 'deepfake,genai'
            }
        )
        return response.json()
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# ----- Streamlit UI -----
st.set_page_config(page_title="DeepShield", page_icon="🕵️", layout="centered")
st.title("DeepShield - Deepfake Detection")

# Image Source Selection
st.markdown("## 🔍 Select Image Source")
source_tab1, source_tab2 = st.tabs(["📁 Upload Image", "🎥 Webcam Capture"])

image = None
analysis_type = st.radio("Analysis Type", ["AI Model", "Sightengine API"], horizontal=True)

with source_tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

with source_tab2:
    ctx = webrtc_streamer(
        key="webcam",
        video_transformer_factory=CaptureImageTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if st.button("📸 Capture Image"):
        if ctx.video_transformer:
            ctx.video_transformer.capture = True
            st.info("Capturing image... Please wait 2 seconds.")
            st.session_state.capture_triggered = True

if 'captured_frame' in st.session_state:
    image = Image.fromarray(st.session_state.captured_frame[..., ::-1])  # BGR to RGB
    del st.session_state.captured_frame

# Display and Analyze Image
if image is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Selected Image", use_column_width=True)
    
    with col2:
        st.markdown("### 🔬 Analysis Results")
        
        if analysis_type == "AI Model":
            image_hash = get_image_hash(image)
            results = predict_image(image_hash, image)
            
            if results:
                chart_data = pd.DataFrame(results)
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x='label',
                    y='score:Q',
                    color='label'
                ).properties(width=400)
                st.altair_chart(chart)
                
                real_score = next(r["score"] for r in results if r["label"] == "real")
                fake_score = next(r["score"] for r in results if r["label"] == "fake")
                st.metric("Real Confidence", f"{real_score*100:.2f}%")
                st.metric("Fake Confidence", f"{fake_score*100:.2f}%")
            else:
                st.error("Prediction failed")
        
        else:  # Sightengine API
            with st.spinner("Analyzing with Sightengine..."):
                result = analyze_with_sightengine(image)
            
            if result:
                if 'error' in result:
                    st.error(f"API Error: {result['error']['message']}")
                else:
                    deepfake_score = result.get('deepfake', 0)
                    ai_score = result.get('ai_generated', 0)
                    
                    st.progress(deepfake_score, text="Deepfake Probability")
                    st.progress(ai_score, text="AI Generation Probability")
                    
                    df = pd.DataFrame({
                        'Type': ['Deepfake', 'AI Generated'],
                        'Score': [deepfake_score, ai_score]
                    })
                    st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
            else:
                st.error("API analysis failed")

# Training Game Section
st.markdown("---")
st.markdown("## 🎮 Training Game")
if st.button("Generate Random Image"):
    if random.choice([True, False]):
        img = fetch_real_image()
        label = "real"
    else:
        img = fetch_fake_image()
        label = "fake"
    
    if img:
        st.session_state.game_image = img
        st.session_state.game_answer = label
    else:
        st.error("Failed to load training images")

if 'game_image' in st.session_state:
    st.image(st.session_state.game_image, width=300)
    guess = st.radio("Is this image real or fake?", ["Real", "Fake"], horizontal=True)
    
    if st.button("Check Answer"):
        user_answer = guess.lower()
        if user_answer == st.session_state.game_answer:
            st.success("Correct! 🎉")
        else:
            st.error(f"Wrong! This is a {st.session_state.game_answer} image")
        del st.session_state.game_image
        del st.session_state.game_answer

# How It Works Section
st.markdown("---")
st.markdown("## 🤖 How It Works")
st.markdown("""
1. **Choose Image Source** - Upload or capture via webcam
2. **Select Analysis Method**:
   - *AI Model*: Local deep learning model
   - *Sightengine API*: Commercial detection service
3. **View Results** - Get confidence scores and visualizations
4. **Training Game** - Practice identifying real/fake images
""")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool provides probabilistic estimates, not absolute determinations. 
Always verify critical content through multiple channels.
""")
