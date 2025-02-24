import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import altair as alt
import io
import hashlib
import random
import os

# ----- Helper functions for fetching images for the game -----
def fetch_real_image():
    """
    Fetch a random real image from the 'game_real' directory.
    Make sure the folder exists and contains image files.
    """
    real_dir = "game_real"
    if not os.path.exists(real_dir):
        st.error("Error: 'game_real' directory not found. Please create it and add image files.")
        return None
        
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not real_images:
        st.error("Error: No images found in 'game_real' directory. Please add some image files.")
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
        st.error(f"Error loading image {selected_image}: {str(e)}")
        return None

def fetch_fake_image():
    """
    Fetch a fake image from the 'Game_Fake' directory.
    This function strictly uses images from that directory.
    """
    fake_dir = "Game_Fake"
    if not os.path.exists(fake_dir):
        st.error("Error: 'Game_Fake' directory not found. Please create it and add fake images.")
        return None

    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not fake_images:
        st.error("Error: No images found in 'Game_Fake' directory. Please add some fake images.")
        return None

    selected_image = random.choice(fake_images)
    try:
        return Image.open(selected_image).copy()
    except Exception as e:
        st.error(f"Error loading fake image {selected_image}: {str(e)}")
        return None
# ----- End of helper functions for fetching images -----

def fetch_real_image():
    # ... [keep original implementation unchanged] ...

def fetch_fake_image():
    # ... [keep original implementation unchanged] ...

def rerun():
    # ... [keep original implementation unchanged] ...

# =======================
# Modern CSS Styling
# =======================
st.set_page_config(page_title="Deepfake Detective", page_icon="🕵️", layout="centered")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
        * { 
            font-family: 'Inter', sans-serif;
            box-sizing: border-box;
        }
        
        .main { 
            background: linear-gradient(152deg, #0a192f 0%, #172a45 100%);
            color: #ffffff;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #6366f1 0%, #8b5cf6 100%);
            color: white !important;
            border-radius: 12px;
            padding: 14px 28px;
            border: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 500;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(99, 102, 241, 0.3);
        }
        
        .glass-panel {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .metric-card {
            background: linear-gradient(45deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }
        
        .stFileUploader>div>div>div>div {
            color: #ffffff;
            border: 2px dashed #6366f1;
            background: rgba(99, 102, 241, 0.05);
            border-radius: 12px;
        }
        
        .game-card {
            transition: all 0.3s ease;
            border-radius: 16px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
            transition: width 0.5s ease;
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# Welcome Page
# =======================
def welcome():
    try:
        logo_image = Image.open("logo.png")
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(logo_image, use_column_width=True)
    except Exception as e:
        st.write("Logo not found.")

    with st.container():
        st.markdown("""
        <div class="glass-panel" style="text-align: center;">
            <h1 style="margin:0; color: #e0e7ff;">DeepVision Analyzer</h1>
            <p style="opacity: 0.8;">Advanced AI-Powered Image Authentication</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            <div class="glass-panel">
                <h3>🔍 Project Overview</h3>
                <p>Our cutting-edge platform combines AI analysis with interactive challenges to detect synthetic media.</p>
                <h3>✨ Key Features</h3>
                <ul style="padding-left: 1.5rem;">
                    <li>Real-time deepfake detection</li>
                    <li>Interactive authentication game</li>
                    <li>Detailed confidence metrics</li>
                    <li>Sample image library</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-panel">
                <h3>🚀 Quick Start</h3>
                <div style="margin: 2rem 0;">
                    <button style="width: 100%; margin: 1rem 0; padding: 12px; border-radius: 8px; 
                            background: rgba(99, 102, 241, 0.1); border: 1px solid #6366f1; color: #e0e7ff;">
                        🧪 Start Image Analysis
                    </button>
                    <button style="width: 100%; margin: 1rem 0; padding: 12px; border-radius: 8px; 
                            background: rgba(139, 92, 246, 0.1); border: 1px solid #8b5cf6; color: #e0e7ff;">
                        🎮 Start Detection Game
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =======================
# Updated Main Page
# =======================
def main():
    try:
        logo_image = Image.open("logo.png")
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(logo_image, use_column_width=True)
    except Exception as e:
        st.write("Logo not found.")

    with st.sidebar:
        st.markdown("""
        <div class="glass-panel">
            <h3 style="margin-top:0;">🔮 Navigation</h3>
            <button style="width: 100%; margin: 0.5rem 0; padding: 12px; border-radius: 8px; 
                    background: rgba(99, 102, 241, 0.1); border: 1px solid #6366f1; color: #e0e7ff;">
                🧪 Image Analysis
            </button>
            <button style="width: 100%; margin: 0.5rem 0; padding: 12px; border-radius: 8px; 
                    background: rgba(139, 92, 246, 0.1); border: 1px solid #8b5cf6; color: #e0e7ff;">
                🎮 Detection Game
            </button>
        </div>
        """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="glass-panel">
            <h3 style="margin-top:0;">📤 Image Analysis Zone</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 3])
        with col1:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            sample_option = st.selectbox("Or try a sample:", ["Select", "Real Sample", "Fake Sample"])
        
        with col2:
            if uploaded_file or sample_option != "Select":
                try:
                    # ... [keep original image loading logic] ...
                    if image:
                        st.image(image, use_column_width=True, caption="Selected Image Preview")
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")

        if (uploaded_file or sample_option != "Select") and 'image' in locals() and image is not None:
            try:
                # ... [keep original analysis logic] ...
                st.markdown("""
                <div class="glass-panel">
                    <h3 style="margin-top:0;">📊 Detection Report</h3>
                """, unsafe_allow_html=True)
                
                # ... [keep original chart logic] ...
                
                # Update result display
                result_style = """
                    background: rgba(255, 255, 255, 0.05);
                    border-left: 4px solid {color};
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1rem 0;
                """
                if final_pred == "fake":
                    st.markdown(f"""
                    <div style="{result_style.format(color='#ff4d4d')}">
                        <h3 style="margin:0; color: #ff4d4d;">🚨 AI Detected! ({scores[final_pred]*100:.2f}%)</h3>
                        <p style="margin:0.5rem 0; opacity:0.8;">This image shows signs of artificial generation</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="{result_style.format(color='#00ff88')}">
                        <h3 style="margin:0; color: #00ff88;">✅ Authentic Content ({scores[final_pred]*100:.2f}%)</h3>
                        <p style="margin:0.5rem 0; opacity:0.8;">No significant AI manipulation detected</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"🔧 Analysis error: {str(e)}")

# =======================
# Enhanced Game Page
# =======================
def game():
    # ... [keep original game logic but add these styling updates] ...
    st.markdown(f"""
    <div class="glass-panel">
        <h2 style="margin:0;">🎮 Deepfake Challenge</h2>
        <div style="display: flex; justify-content: space-between; margin: 2rem 0;">
            <div class="metric-card">
                <h4>Round</h4>
                <h1>{st.session_state.game_round}/5</h1>
            </div>
            <div class="metric-card">
                <h4>Score</h4>
                <h1>{st.session_state.game_score}</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ... [rest of game page logic with updated styling] ...

# =======================
# Page Routing
# =======================
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
        
    if st.session_state.page == "game":
        game()
    elif st.session_state.page == "main":
        main()
    else:
        welcome()
