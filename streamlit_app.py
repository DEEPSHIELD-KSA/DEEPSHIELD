import streamlit as st
from PIL import Image
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
        
        .analysis-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            animation: cardEntrance 0.6s ease-out;
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
        
        .glow {
            animation: glow 2s ease-in-out infinite;
        }
        
        @keyframes glow {
            0% { filter: drop-shadow(0 0 5px #00bcd4); }
            50% { filter: drop-shadow(0 0 20px #00bcd4); }
            100% { filter: drop-shadow(0 0 5px #00bcd4); }
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
    <div class="analysis-card">
        <h3>🕵️ Advanced Deepfake Detection</h3>
        <p>Combining cutting-edge AI with professional analysis</p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 2rem;">
            <div class="analysis-card">
                <h4>📸 Instant Analysis</h4>
                <p>Real-time deepfake detection</p>
            </div>
            <div class="analysis-card">
                <h4>🔐 Secure Processing</h4>
                <p>Military-grade encryption</p>
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

    with col2:
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Image Error: {str(e)}")

    if uploaded_file and 'image' in locals():
        try:
            with st.spinner("🔍 Analyzing with Professional API..."):
                image_bytes = uploaded_file.getvalue()
                api_results = analyze_with_sightengine(image_bytes)
                
            if api_results:
                st.markdown("## 🔬 Analysis Report")
                conclusion = (
                    "❌ Confirmed Deepfake" if api_results['deepfake'] > 85 else
                    "🤖 AI-Generated Content" if api_results['ai_generated'] > 85 else
                    "✅ Authentic Content"
                )
                
                st.markdown(f"""
                <div class="analysis-card glow">
                    <div style="text-align: center; padding: 2rem;">
                        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">
                            {conclusion.split()[0]} {''.join(conclusion.split()[1:])}
                        </h1>
                        <div class="confidence-meter">
                            <div class="meter-bar" style="width: {max(api_results['deepfake'], api_results['ai_generated']):.1f}%">
                                <span class="meter-text">
                                    {max(api_results['deepfake'], api_results['ai_generated']):.1f}% Confidence
                                </span>
                            </div>
                        </div>
                        <div style="margin-top: 2rem; display: grid; gap: 1rem;">
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 15px;">
                                <h4>🧪 Deepfake Probability</h4>
                                <h2>{api_results['deepfake']:.1f}%</h2>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 15px;">
                                <h4>🧠 AI-Generated Probability</h4>
                                <h2>{api_results['ai_generated']:.1f}%</h2>
                            </div>
                        </div>
                    </div>
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

    with st.sidebar:
        if st.button("← Return to Main"):
            st.session_state.page = "main"
            st.rerun()

    # Game over condition
    if st.session_state.game_round > 5:
        st.markdown(f"""
        <div class="analysis-card" style="text-align: center;">
            <h2>Game Over! 🎯</h2>
            <h1 style="color: #00bcd4; font-size: 3rem;">{st.session_state.game_score}/5</h1>
            <div style="margin-top: 2rem;">
                <button class="restart-btn" onclick="window.location.reload()">Play Again 🔄</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Round display
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; margin-bottom: 2rem;">
        <h3>Round {st.session_state.game_round} of 5</h3>
        <div style="background: #00bcd4; padding: 0.5rem 1rem; border-radius: 25px;">
            Score: {st.session_state.game_score}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load images
    if "current_round" not in st.session_state:
        try:
            real_img = fetch_real_image()
            fake_img = fetch_fake_image()
            if real_img and fake_img:
                st.session_state.current_round = {
                    "images": random.sample([(real_img, "Real"), (fake_img, "Fake")], 2),
                    "answer": random.choice(["Left", "Right"]),
                    "images_base64": [image_to_base64(real_img), image_to_base64(fake_img)]
                }
            else:
                st.error("Failed to load game images")
                return
        except Exception as e:
            st.error(f"Game initialization error: {str(e)}")
            return

    # Display images
    if "current_round" in st.session_state:
        cols = st.columns(2)
        for idx in range(2):
            with cols[idx]:
                st.markdown(f"""
                <div class="game-image-container">
                    <img src="data:image/png;base64,{st.session_state.current_round['images_base64'][idx]}" 
                        class="game-image">
                </div>
                """, unsafe_allow_html=True)

        user_guess = st.radio("Which image is real?", ["Left", "Right"], horizontal=True)
        
        if st.button("Submit Answer", key="guess_btn"):
            if user_guess == st.session_state.current_round["answer"]:
                st.session_state.game_score += 1
                st.success("Correct! 🎉 +1 Point")
            else:
                st.error("Incorrect ❌ Try again!")
            
            del st.session_state.current_round
            st.session_state.game_round += 1
            st.experimental_rerun()

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
