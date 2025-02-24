import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import altair as alt
import io
import hashlib
import random
import time  # Added for automatic progression

# Set page config with new name and logo
st.set_page_config(page_title="DEEPSHIELD", page_icon="🛡️", layout="centered")

# Custom CSS with updated branding
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&display=swap');
    
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }
    
    .stButton>button {
        background: #2196F3;
        color: white;
        border-radius: 15px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #64B5F6;
        transform: scale(1.05);
    }
    
    .header-logo {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# Load the deepfake detection model
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

# =======================
# Main Page: Image Analysis
# =======================
def main():
    # Display logo
    st.markdown('<div class="header-logo">', unsafe_allow_html=True)
    st.image("logo.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.title("DEEPSHIELD AI Detection System")
    
    # Sidebar with Navigation
    with st.sidebar:
        st.markdown("""
            <div style="border-left: 3px solid #2196F3; padding-left: 1rem; margin: 1rem 0;">
                <h2 style="color: #2196F3;">🛡️ Navigation</h2>
                <p>Test your skills in our detection challenge!</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🎮 Start Detection Game", use_container_width=True):
            st.session_state.page = "game"
            st.rerun()
    
    # Rest of main page content remains similar with updated styling...

# =======================
# Updated Game Page with Auto-Advance
# =======================
def game():
    # Display logo
    st.markdown('<div class="header-logo">', unsafe_allow_html=True)
    st.image("logo.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.title("DEEPSHIELD Challenge")
    
    # Initialize game state
    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_round" not in st.session_state:
        st.session_state.game_round = 1
    if "show_result" not in st.session_state:
        st.session_state.show_result = False
    if "current_round" not in st.session_state:
        st.session_state.current_round = None

    # Game over state
    if st.session_state.game_round > 5:
        st.balloons()
        st.markdown(f"## Game Over! Final Score: {st.session_state.game_score}/5")
        if st.button("Play Again"):
            st.session_state.clear()
            st.rerun()
        return

    # Round progress
    st.markdown(f"### Round {st.session_state.game_round}/5")
    progress = st.progress(st.session_state.game_round * 0.2)

    # Game logic
    if not st.session_state.current_round or st.session_state.show_result:
        # Initialize new round
        real_image = Image.open("samples/real_sample.jpg")  # Replace with your real images
        fake_image = Image.open("samples/fake_sample.jpg")  # Replace with your fake images
        
        # Randomize positions
        if random.choice([True, False]):
            st.session_state.current_round = {
                "left": real_image,
                "right": fake_image,
                "answer": "left"
            }
        else:
            st.session_state.current_round = {
                "left": fake_image,
                "right": real_image,
                "answer": "right"
            }
        st.session_state.show_result = False

    # Display images
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.current_round["left"], use_container_width=True)
    with col2:
        st.image(st.session_state.current_round["right"], use_container_width=True)

    # User input and automatic progression
    if not st.session_state.show_result:
        user_choice = st.radio("Which image is real?", ["left", "right"])
        if st.button("Submit"):
            if user_choice == st.session_state.current_round["answer"]:
                st.session_state.game_score += 1
                st.success("Correct! 🎉")
            else:
                st.error("Wrong! 😢")
            
            st.session_state.show_result = True
            time.sleep(1.5)  # Show result for 1.5 seconds
            st.session_state.game_round += 1
            st.rerun()

# =======================
# Page Routing
# =======================
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main()
elif st.session_state.page == "game":
    game()
