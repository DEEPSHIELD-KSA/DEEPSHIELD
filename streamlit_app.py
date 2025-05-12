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
import base64
from keras import applications
import requests

# Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"

# ----- Constants & Configurations -----
API_USER = "1285106646"
API_KEY = "CDWtk3q6HdqHcs6DJxn9Y8YnL46kz6pX"

# ----- Helper Functions -----
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

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
    if "used_fake_images" not in st.session_state:
        st.session_state.used_fake_images = set()
    available_images = [img for img in fake_images if img not in st.session_state.used_fake_images]
    if not available_images:
        st.session_state.used_fake_images = set()
        available_images = fake_images
    selected_image = random.choice(available_images)
    st.session_state.used_fake_images.add(selected_image)
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
def analyze_with_sightengine(image_bytes):
    try:
        response = requests.post(
            'https://api.sightengine.com/1.0/check.json',
            files={'media': ('image.jpg', image_bytes, 'image/jpeg')},
            data={
                'api_user': API_USER,
                'api_secret': API_KEY,
                'models': 'deepfake,genai'
            }
        )
        result = response.json()
        scores = {
            'deepfake': result['type'].get('deepfake', 0.0) * 100,
            'ai_generated': result['type'].get('ai_generated', 0.0) * 100
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
        :root {
            --primary: #00bcd4;
            --secondary: #001f3f;
            --accent: #ff4d4d;
        }
        
        .main { 
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%); 
            color: white;
        }
        
        .stButton>button {
            background: var(--primary);
            border: 2px solid white;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background: var(--secondary);
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .game-image-container {
            width: 400px;
            height: 400px;
            border-radius: 20px;
            overflow: hidden;
            margin: 0 auto 2rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            position: relative;
        }
        
        .game-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }
        
        .game-image:hover {
            transform: scale(1.05);
        }
        
        .round-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
        }
        
        .round-text {
            font-size: 1.2rem;
            color: #00bcd4;
        }
        
        .score-badge {
            background: #00bcd4;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: bold;
        }
        
        .conclusion-card {
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            animation: cardEntrance 0.6s ease-out;
        }
        
        .ai-generated {
            background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(255,142,83,0.15));
            border-color: #ff6b6b;
        }
        
        .conclusion-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        .conclusion-icon {
            font-size: 2.5rem;
            animation: iconFloat 3s ease-in-out infinite;
        }
        
        .confidence-meter {
            height: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
            position: relative;
            margin: 1.5rem 0;
        }
        
        .meter-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), #ff8e53);
            position: relative;
            transition: width 1s ease-out;
        }
        
        @keyframes cardEntrance {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes iconFloat {
            0% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

def welcome_page():
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            st.image(Image.open("logo.png"), width=200)
        except:
            pass
    with col2:
        st.title("DeepShield AI Detector")
    
    st.markdown("""
    <div class="metric-card">
        <h3>🕵️ Advanced Deepfake Detection</h3>
        <p>Combining cutting-edge AI models with professional API analysis</p>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 2rem;">
            <div class="metric-card">
                <h4>📸 Image Analysis</h4>
                <p>Dual detection systems for maximum accuracy</p>
            </div>
            
            <div class="metric-card">
                <h4>🔐 Secure Processing</h4>
                <p>Military-grade encryption for all uploads</p>
            </div>
            
            <div class="metric-card">
                <h4>🤖 AI-Powered</h4>
                <p>State-of-the-art neural networks</p>
            </div>
            
            <div class="metric-card">
                <h4>📊 Detailed Reports</h4>
                <p>Comprehensive analysis results</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Detection →", key="start_btn"):
        st.session_state.page = "main"
        st.rerun()

# ----- Main Detection Interface -----
def main_interface():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        detection_mode = st.radio("Detection Mode", ["API Analysis", "Local Model"])
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
            if detection_mode == "API Analysis":
                with st.spinner("🔍 Analyzing with Professional API..."):
                    image_bytes = uploaded_file.getvalue() if uploaded_file else (
                        open("samples/real_sample.jpg" if sample_option == "Real Sample" 
                            else "samples/fake_sample.jpg", "rb").read())
                    api_results = analyze_with_sightengine(image_bytes)
                    
                if api_results:
                    st.markdown("## 🔬 Professional Analysis Report")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🧪 Deepfake Probability</h4>
                            <h1 style="color: var(--accent);">{api_results['deepfake']:.0f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>🧠 AI-Generated Probability</h4>
                            <h1 style="color: var(--accent);">{api_results['ai_generated']:.0f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    conclusion = (
                        "❌ Confirmed Deepfake" if api_results['deepfake'] > 85 else
                        "🤖 AI-Generated Content" if api_results['ai_generated'] > 85 else
                        "✅ Authentic Content"
                    )
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🔍 Expert Analysis Conclusion</h3>
                        <div class="conclusion-card {'ai-generated' if '🤖' in conclusion else ''}">
                            <div class="conclusion-header">
                                <span class="conclusion-icon">
                                    {"🤖" if '🤖' in conclusion else "❌" if '❌' in conclusion else "✅"}
                                </span>
                                <h2 class="conclusion-title">{conclusion}</h2>
                            </div>
                            
                            <div class="confidence-meter">
                                <div class="meter-bar" style="width: {max(api_results['deepfake'], api_results['ai_generated']) if '🤖' in conclusion else 100 - max(api_results['deepfake'], api_results['ai_generated'])}%">
                                    <span class="meter-text">
                                        {max(api_results['deepfake'], api_results['ai_generated']) if '🤖' in conclusion else 100 - max(api_results['deepfake'], api_results['ai_generated'])}% Confidence
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                with st.spinner("🤖 Analyzing with Local AI Model..."):
                    image_hash = get_image_hash(image)
                    model_results = predict_image(image_hash, image)
                    scores = {r["label"]: r["score"] for r in model_results}
                    
                st.markdown("## 📊 Local Model Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>✅ Real Confidence</h4>
                        <h1 style="color: #00ff88;">{scores['real']*100:.0f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>❌ Fake Confidence</h4>
                        <h1 style="color: var(--accent);">{scores['fake']*100:.0f}%</h1>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

# ----- Game Interface -----
def game_interface():
    st.title("🎮 Detection Training")
    
    # Initialize game state
    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_round" not in st.session_state:
        st.session_state.game_round = 1

    # Add return button to sidebar
    with st.sidebar:
        if st.button("← Return to Main"):
            st.session_state.page = "main"
            st.rerun()

    # Game over condition
    if st.session_state.game_round > 5:
        st.markdown(f"""
        <div class="metric-card" style="text-align: center;">
            <h2>Game Over! 🎯</h2>
            <h1 style="color: #00bcd4; font-size: 3rem;">Final Score: {st.session_state.game_score}/5</h1>
            <div style="margin-top: 2rem;">
                <button class="restart-btn" onclick="window.location.reload()">Play Again 🔄</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Round display
    st.markdown(f"""
    <div class="round-header">
        <span class="round-text">Round {st.session_state.game_round} of 5</span>
        <span class="score-badge">Score: {st.session_state.game_score}</span>
    </div>
    """, unsafe_allow_html=True)

    # Load images with caching
    if "current_round" not in st.session_state:
        try:
            real_img = fetch_real_image()
            fake_img = fetch_fake_image()
            if real_img and fake_img:
                # Pre-resize images for better performance
                real_img = real_img.resize((400, 400))
                fake_img = fake_img.resize((400, 400))
                
                st.session_state.current_round = {
                    "images": random.sample([
                        (real_img, "Real"), 
                        (fake_img, "Fake")
                    ], 2),
                    "answer": random.choice(["Left", "Right"]),
                    "images_base64": [
                        image_to_base64(real_img),
                        image_to_base64(fake_img)
                    ]
                }
            else:
                st.error("Failed to load game images")
                return
        except Exception as e:
            st.error(f"Game initialization error: {str(e)}")
            return

    # Display images from cached base64
    if "current_round" in st.session_state:
        cols = st.columns(2)
        for idx in range(2):
            with cols[idx]:
                st.markdown(f"""
                <div class="game-card">
                    <div class="game-image-container">
                        <img src="data:image/png;base64,{st.session_state.current_round['images_base64'][idx]}" 
                            class="game-image">
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # User input
        user_guess = st.radio("Which image is real?", ["Left", "Right"], horizontal=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Submit Answer", key="guess_btn"):
                if user_guess == st.session_state.current_round["answer"]:
                    st.session_state.game_score += 1
                    st.success("Correct! 🎉 +1 Point")
                else:
                    st.error("Incorrect ❌ Try again!")
                
                # Clear current round and advance
                del st.session_state.current_round
                st.session_state.game_round += 1
                st.experimental_rerun()
                
        with col2:
            if st.button("↩ Return to Main Analysis"):
                st.session_state.page = "main"
                st.experimental_rerun()

# Update confidence meter in main interface
# In the conclusion section replace the confidence meter with:
st.markdown(f"""
<div class="confidence-meter">
    <div class="meter-bar" style="width: {max(api_results['deepfake'], api_results['ai_generated']) if '🤖' in conclusion else 100 - max(api_results['deepfake'], api_results['ai_generated']):.1f}%">
        <span class="meter-text">
            {max(api_results['deepfake'], api_results['ai_generated']) if '🤖' in conclusion else 100 - max(api_results['deepfake'], api_results['ai_generated']):.1f}% Confidence
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
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
