import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import altair as alt
import requests
import os
import random
import time  # For automatic round progression

# Custom CSS with Blue Theme
st.markdown("""
    <style>
    :root {
        --primary: #2196F3;
        --bg: #1a237e;
        --secondary: #0D47A1;
        --accent: #64B5F6;
        --text: #E3F2FD;
    }
    
    .main {
        background: linear-gradient(135deg, var(--bg) 0%, var(--secondary) 100%);
        color: var(--text);
    }
    
    .stButton>button {
        background: var(--primary);
        color: white;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .header-logo {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# Page Config
st.set_page_config(page_title="DEEPSHIELD", page_icon="🛡️", layout="centered")

# Load the deepfake detection model
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

# ... [Keep existing fetch_fake_image and fetch_real_image functions] ...

# Main Page
def main():
    # Display logo
    st.markdown('<div class="header-logo">', unsafe_allow_html=True)
    st.image("logo.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.title("DEEPSHIELD AI Detection System")
    
    # Rest of main page content...
    # [Keep your existing main page implementation here]

# Modified Game Page with Auto-Advance
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

    # Game over state
    if st.session_state.game_round > 5:
        st.balloons()
        st.write(f"## Game Over! Final Score: {st.session_state.game_score}/5")
        if st.button("Play Again"):
            st.session_state.clear()
            st.rerun()
        return

    # Round progress
    st.write(f"### Round {st.session_state.game_round}/5")
    progress = st.session_state.game_round * 20
    st.progress(progress)

    # Game logic
    if "current_round" not in st.session_state or st.session_state.show_result:
        # Initialize new round
        real_image = fetch_real_image()
        fake_image = fetch_fake_image()
        
        if real_image and fake_image:
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

    # User input
    if not st.session_state.show_result:
        user_choice = st.radio("Which image is real?", ["left", "right"])
        if st.button("Submit"):
            if user_choice == st.session_state.current_round["answer"]:
                st.session_state.game_score += 1
                st.success("Correct! 🎉")
            else:
                st.error("Wrong! 😢")
            
            st.session_state.show_result = True
            time.sleep(2)  # Show result for 2 seconds
            st.session_state.game_round += 1
            st.rerun()

# Page routing
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main()
elif st.session_state.page == "game":
    game()
