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

# ----- End of helper functions -----

# Helper function for page reruns
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# Set page config
st.set_page_config(page_title="Deepfake", page_icon="🕵️", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&display=swap');
        * { font-family: 'Space+Grotesk', sans-serif; }
        .main { background: linear-gradient(135deg, #001f3f 0%, #00bcd4 100%); color: #ffffff; }
        .stButton>button { background: #00bcd4; color: white; border-radius: 15px; padding: 10px 24px; border: none; transition: all 0.3s ease; }
        .stButton>button:hover { background: #008ba3; transform: scale(1.05); }
        .stFileUploader>div>div>div>div { color: #ffffff; border: 2px dashed #00bcd4; background: rgba(0, 188, 212, 0.1); border-radius: 15px; }
        .metric-box { background: rgba(0, 188, 212, 0.1); padding: 20px; border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .game-image { border: 3px solid transparent; border-radius: 15px; transition: all 0.3s ease; }
        .game-image:hover { transform: scale(1.02); cursor: pointer; }
        .welcome-section { background: rgba(255, 255, 255, 0.05); padding: 2rem; border-radius: 15px; margin: 1rem 0; }
        .made-by-box { 
            background: rgba(0,188,212,0.15); 
            border: 2px solid #00bcd4; 
            padding: 20px; 
            border-radius: 0px; 
            margin: 10px auto; 
            text-align: center; 
            width: 100%; 
            max-width: 500px; 
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
        }
        .made-by-text { 
            font-weight: bold; 
            color: #ffffff; 
            margin: 0; 
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# Welcome Page
# =======================
def welcome():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=700)
    except Exception as e:
        st.write("")

    st.title("DeepShield")

    st.markdown("""
    <div class="welcome-section">
        <h3>Welcome to the Deepfake Detection final project</h3>
        <p>This application helps you detect AI-generated images using state-of-the-art machine learning models. 
        Explore the capabilities of deepfake detection through image analysis or test your skill in the detection challenge game!</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])  # Centering the Made By section
    with col3:
        st.markdown("""
        <div class="made-by-box">
            <h3 class="made-by-text">🕵️ Made by:</h3>
            <p class="made-by-text">Musab Alosaimi - Bassam Alanazi - Abdulazlz AlHwitan</p>
        </div>
        """, unsafe_allow_html=True)

    if st.button("Get Started →", use_container_width=True):
        st.session_state.page = "main"
        rerun()

# =======================
# Page Routing
# =======================
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        welcome()
