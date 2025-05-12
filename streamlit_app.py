import streamlit as st
from PIL import Image
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

# ----- Constants -----
API_USER = "1285106646"
API_KEY = "CDWtk3q6HdqHcs6DJxn9Y8YnL46kz6pX"

# ----- Image Fetching Functions -----
def fetch_real_image():
    # ... (keep the same as previous version) ...

def fetch_fake_image():
    # ... (keep the same as previous version) ...

# ----- AI Detection Functions -----
@st.cache_resource
def load_model():
    # ... (keep the same as previous version) ...

def get_image_hash(image: Image.Image) -> str:
    # ... (keep the same as previous version) ...

@st.cache_data(show_spinner=False)
def predict_image(image_hash: str, _image: Image.Image):
    # ... (keep the same as previous version) ...

# ----- Sightengine API Integration -----
def analyze_with_sightengine(image_bytes):
    # ... (keep the same as previous version) ...

# ----- UI Components -----
def setup_page():
    st.set_page_config(page_title="DeepShield", page_icon="🕵️", layout="centered")
    st.markdown("""
    <style>
        :root {
            --primary: #00bcd4;
            --secondary: #001f3f;
            --accent: #ff4d4d;
            --success: #00ff88;
        }
        
        .main { 
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
            color: white;
        }
        
        .stButton>button {
            background: var(--primary);
            border: 2px solid white;
            border-radius: 25px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .analysis-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            margin: 1rem 0;
        }
        
        .game-image-container {
            width: 100%;
            height: 300px;
            border-radius: 20px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        @media (max-width: 768px) {
            .game-image-container {
                height: 200px;
            }
            .stButton>button {
                padding: 0.5rem;
                font-size: 14px;
            }
        }
        
        .result-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .score-board {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background: rgba(0, 188, 212, 0.2);
            border-radius: 15px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def welcome_page():
    # ... (keep the same as previous version but ensure logo is responsive) ...
    try:
        st.image(Image.open("logo.png"), use_column_width=True)
    except:
        pass

# ----- Main Detection Interface -----
def main_interface():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        detection_mode = st.radio("Detection Mode", ["API Analysis", "Local Model"])
        if st.button("← Return to Welcome"):
            st.session_state.page = "welcome"
            st.rerun()

    # ... (keep file uploader and sample selection same) ...

    if (uploaded_file or sample_option != "Select") and 'image' in locals():
        try:
            if detection_mode == "API Analysis":
                # ... (keep API analysis logic) ...
                
                if api_results:
                    st.markdown("## 🔬 Analysis Report")
                    conclusion = (
                        "❌ Deepfake Detected" if api_results['deepfake'] > 40 else
                        "🤖 AI-Generated" if api_results['ai_generated'] > 40 else
                        "✅ Authentic Content"
                    )
                    
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div style="text-align: center;">
                            <div class="result-icon">
                                {"❌" if '❌' in conclusion else "🤖" if '🤖' in conclusion else "✅"}
                            </div>
                            <h2>{conclusion}</h2>
                            <div class="score-board">
                                <div>
                                    <h4>Deepfake</h4>
                                    <h3>{api_results['deepfake']:.1f}%</h3>
                                </div>
                                <div>
                                    <h4>AI Generated</h4>
                                    <h3>{api_results['ai_generated']:.1f}%</h3>
                                </div>
                            </div>
                            <div class="confidence-bar">
                                <div style="width: {max(api_results['deepfake'], api_results['ai_generated'])}%; 
                                    height: 100%; 
                                    background: {'var(--accent)' if '❌' in conclusion or '🤖' in conclusion else 'var(--success)'};
                                    transition: width 0.5s ease;">
                                </div>
                            </div>
                            <p>Overall Confidence: {max(api_results['deepfake'], api_results['ai_generated']):.1f}%</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:  # Local Model
                # ... (keep local model logic) ...
                
                st.markdown("## 📊 Local Analysis")
                st.markdown(f"""
                <div class="analysis-card">
                    <div style="text-align: center;">
                        <div class="result-icon">
                            {"❌" if scores['fake'] > 0.5 else "✅"}
                        </div>
                        <h2>{'❌ Fake Content Detected' if scores['fake'] > 0.5 else '✅ Authentic Content'}</h2>
                        <div class="score-board">
                            <div>
                                <h4>Real Confidence</h4>
                                <h3>{scores['real']*100:.1f}%</h3>
                            </div>
                            <div>
                                <h4>Fake Confidence</h4>
                                <h3>{scores['fake']*100:.1f}%</h3>
                            </div>
                        </div>
                        <div class="confidence-bar">
                            <div style="width: {max(scores['real'], scores['fake'])*100}%; 
                                height: 100%; 
                                background: {'var(--accent)' if scores['fake'] > 0.5 else 'var(--success)'};
                                transition: width 0.5s ease;">
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
    
    with st.sidebar:
        if st.button("← Return to Main"):
            st.session_state.page = "main"
            st.rerun()

    if st.session_state.get("game_round", 1) > 5:
        st.markdown(f"""
        <div class="analysis-card" style="text-align: center;">
            <h2>Game Over! 🎯</h2>
            <h1 style="color: var(--primary);">{st.session_state.game_score}/5</h1>
            <div style="display: grid; gap: 1rem; margin-top: 2rem;">
                <button class="stButton" onclick="window.location.reload()">Play Again 🔄</button>
                <button class="stButton" onclick="window.location.href='/'">Return Home 🏠</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="score-board">
        <span>Round {st.session_state.game_round}/5</span>
        <span>Score: {st.session_state.game_score}</span>
    </div>
    """, unsafe_allow_html=True)

    # ... (keep game image loading logic) ...

    if "current_round" in st.session_state:
        cols = st.columns(2)
        for idx in range(2):
            with cols[idx]:
                st.markdown(f"""
                <div class="game-image-container">
                    <img src="data:image/png;base64,{st.session_state.current_round['images_base64'][idx]}" 
                        style="width: 100%; height: 100%; object-fit: cover;">
                </div>
                """, unsafe_allow_html=True)

        user_guess = st.radio("Which image is real?", ["Left", "Right"], horizontal=True)
        
        if st.button("Submit Answer"):
            # ... (keep answer handling logic) ...

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
