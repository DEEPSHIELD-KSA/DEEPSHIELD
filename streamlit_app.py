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

# Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"

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

# ----- UI Components -----
def setup_page():
    st.set_page_config(page_title="DeepShield", page_icon="🕵️", layout="centered")
    st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #001f3f 0%, #00bcd4 100%); color: white; }
        .stButton>button { background: #00bcd4; border-radius: 15px; transition: 0.3s; }
        .stButton>button:hover { background: #008ba3; transform: scale(1.05); }
        .metric-box { background: rgba(0, 188, 212, 0.1); border-radius: 15px; padding: 20px; }
        .game-image { border-radius: 15px; transition: 0.3s; }
        .game-image:hover { transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

def welcome_page():
    try:
        st.image(Image.open("logo.png"), width=800)
    except:
        pass
    st.title("DeepShield AI Detector")
    st.markdown("""
    <div class="metric-box">
        <h3>🕵️ Detect Deepfakes & AI-Generated Content</h3>
        <p>Combine local AI models with Sightengine API for comprehensive analysis</p>
        <h4>Features:</h4>
        <li>📸 Image analysis with dual detection systems</li>
        <li>🔐 Secure API integration</li>
        <li>🎮 Interactive detection game</li>
        <li>📊 Detailed confidence metrics</li>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start Detection →"):
        st.session_state.page = "main"
        st.rerun()

# ----- Main Detection Interface -----
def main_interface():
    with st.sidebar:
        st.markdown("## 🔐 API Settings")
        api_user = st.text_input("Sightengine API User")
        api_secret = st.text_input("Sightengine API Secret", type="password")
        st.markdown("---")
        st.markdown("## 🎮 Game Controls")
        if st.button("New Detection Game"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.session_state.page = "game"
            st.rerun()
        if st.button("← Return to Welcome"):
            st.session_state.page = "welcome"
            st.rerun()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("## 📤 Image Analysis")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        sample_option = st.selectbox("Or choose sample:", ["Select", "Real Sample", "Fake Sample"])

    with col2:
        if uploaded_file or sample_option != "Select":
            try:
                image = Image.open(uploaded_file) if uploaded_file else (
                    Image.open("samples/real_sample.jpg") if sample_option == "Real Sample" 
                    else Image.open("samples/fake_sample.jpg"))
                st.image(image, caption="Selected Image", use_container_width=True)
            except Exception as e:
                st.error(f"Image Error: {str(e)}")

    if (uploaded_file or sample_option != "Select") and 'image' in locals():
        try:
            if api_user and api_secret:
                with st.spinner("🔍 Analyzing with Sightengine API..."):
                    image_bytes = uploaded_file.getvalue() if uploaded_file else (
                        open("samples/real_sample.jpg" if sample_option == "Real Sample" 
                            else "samples/fake_sample.jpg", "rb").read())
                    api_results = analyze_with_sightengine(image_bytes, api_user, api_secret)
                    api_results = api_results*100
                    
                if api_results:
                    st.markdown("## 🔬 API Analysis Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>🧪 Deepfake Score</h4>
                            <h2>{api_results['deepfake']:.3f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <h4>🧠 AI-Generated Score</h4>
                            <h2>{api_results['ai_generated']:.3f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    conclusion = (
                        "❌ Deepfake Detected" if api_results['deepfake'] > 40 else
                        "🤖 AI-Generated" if api_results['ai_generated'] > 40 else
                        "✅ Authentic Image"
                    )
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>📝 Conclusion</h3>
                        <h4>{conclusion}</h4>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                with st.spinner("🤖 Analyzing with Local Model..."):
                    image_hash = get_image_hash(image)
                    model_results = predict_image(image_hash, image)
                    scores = {r["label"]: r["score"] for r in model_results}
                    
                st.markdown("## 📊 Local Model Results")
                st.markdown(f"""
                <div class="metric-box">
                    <div style="background: linear-gradient(90deg, #00ff88 {scores['real']*100}%, 
                        rgba(0,0,0,0.1) {scores['real']*100}%); border-radius: 10px; padding: 15px;">
                        <h4>✅ Real Confidence: {scores['real']*100:.2f}%</h4>
                    </div>
                    <div style="margin-top: 20px; background: linear-gradient(90deg, #ff4d4d {scores['fake']*100}%, 
                        rgba(0,0,0,0.1) {scores['fake']*100}%); border-radius: 10px; padding: 15px;">
                        <h4>❌ Fake Confidence: {scores['fake']*100:.2f}%</h4>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

# ----- Game Interface -----
def game_interface():
    st.title("🎮 Detection Game")
    if st.session_state.get("game_round", 1) > 5:
        st.markdown(f"## Game Over! Final Score: {st.session_state.get('game_score', 0)}/5")
        if st.button("Play Again"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.rerun()
        return

    st.markdown(f"### Round {st.session_state.get('game_round', 1)} of 5")
    st.markdown(f"Current Score: {st.session_state.get('game_score', 0)}")

    if "current_round" not in st.session_state:
        real_img = fetch_real_image()
        fake_img = fetch_fake_image()
        if real_img and fake_img:
            st.session_state.current_round = {
                "images": random.sample([(real_img, "Real"), (fake_img, "Fake")], 2),
                "answer": random.choice(["Left", "Right"])
            }

    if "current_round" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.current_round["images"][0][0], 
                    caption="Left Image", use_column_width=True)
        with col2:
            st.image(st.session_state.current_round["images"][1][0], 
                    caption="Right Image", use_column_width=True)

        user_guess = st.radio("Which image is real?", ["Left", "Right"], horizontal=True)
        if st.button("Submit Guess"):
            if user_guess == st.session_state.current_round["answer"]:
                st.session_state.game_score += 1
                st.success("Correct! 🎉")
            else:
                st.error("Wrong Answer 😢")
            st.session_state.game_round += 1
            del st.session_state.current_round
            st.rerun()

# ----- App Flow -----
if __name__ == "__main__":
    setup_page()
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    
    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "game":
        game_interface()
    else:
        main_interface()
