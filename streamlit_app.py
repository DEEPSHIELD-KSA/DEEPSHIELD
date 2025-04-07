# Available backend options are: "jax", "torch", "tensorflow"
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
import io
import hashlib
import random
import numpy as np

# ----- Helper functions for fetching images for the game -----
def fetch_real_image():
    real_dir = "game_real"
    if not os.path.exists(real_dir):
        st.error("Error: 'game_real' directory not found.")
        return None

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not real_images:
        st.error("Error: No images found in 'game_real' directory.")
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
        st.error("Error: 'Game_Fake' directory not found.")
        return None

    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not fake_images:
        st.error("Error: No fake images found.")
        return None

    selected_image = random.choice(fake_images)
    try:
        return Image.open(selected_image).copy()
    except Exception as e:
        st.error(f"Error loading fake image {selected_image}: {str(e)}")
        return None

# ----- Model loading and prediction functions -----
@st.cache_resource
def load_model():
    return keras.saving.load_model("hf://musabalosimi/deepfake1")

def get_image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()

@st.cache_data(show_spinner=False)
def predict_image(image_hash: str, _image: Image.Image):
    model = load_model()
    
    # Preprocess the image
    img = _image.convert('RGB').resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get prediction
    prediction = model.predict(img_array)
    prob = prediction[0][0]
    
    # Format results
    if prob > 0.5:
        return [{"label": "real", "score": float(prob)}, {"label": "fake", "score": float(1 - prob)}]
    else:
        return [{"label": "fake", "score": float(1 - prob)}, {"label": "real", "score": float(prob)}]

# ----- UI Components and Page Config -----
st.set_page_config(page_title="Deepfake", page_icon="🕵️", layout="centered")

st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #001f3f 0%, #00bcd4 100%); color: #ffffff; }
        .stButton>button { background: #00bcd4; border-radius: 15px; }
        .metric-box { background: rgba(0, 188, 212, 0.1); padding: 20px; border-radius: 15px; }
        .game-image { border-radius: 15px; transition: all 0.3s ease; }
        .welcome-section { background: rgba(255, 255, 255, 0.05); padding: 2rem; border-radius: 15px; }
    </style>
""", unsafe_allow_html=True)

# ----- Page Functions -----
def welcome():
    st.title("DeepShield")
    st.markdown("""
    <div class="welcome-section">
        <h3>Welcome to Deepfake Detection</h3>
        <p>Detect AI-generated images using state-of-the-art models or test your skills in our detection game!</p>
        <h3>🕵️ Made By</h3>
        <li>Musab Alosaimi</li>
        <li>Bassam Alanazi</li>
        <li>Abdulaziz AlHwitan</li>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Get Started →"):
        st.session_state.page = "main"
        st.rerun()

def main():
    st.title("Deepfake Detection System")

    with st.sidebar:
        if st.button("🏠 Return to Welcome"):
            st.session_state.page = "welcome"
            st.rerun()
        if st.button("🎮 Start Detection Game"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.session_state.used_real_images = set()
            st.session_state.page = "game"
            st.rerun()

    col1, col2 = st.columns([4, 3])
    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        sample_option = st.selectbox("Or choose samples:", ["Select", "Real Sample", "Fake Sample"])

    if uploaded_file or sample_option != "Select":
        image = None
        if uploaded_file:
            image = Image.open(uploaded_file)
        elif sample_option == "Real Sample":
            image = Image.open("samples/real_sample.jpg") if os.path.exists("samples/real_sample.jpg") else None
        elif sample_option == "Fake Sample":
            image = Image.open("samples/fake_sample.jpg") if os.path.exists("samples/fake_sample.jpg") else None

        if image:
            st.image(image, use_container_width=True)
            with st.spinner("Analyzing..."):
                image_hash = get_image_hash(image)
                result = predict_image(image_hash, image)
                scores = {r["label"].lower(): r["score"] for r in result}

            st.markdown("### 📊 Results")
            real_conf = scores.get("real", 0)
            fake_conf = scores.get("fake", 0)
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Real Confidence</h3>
                    <h2 style="color: #00ff88;">{real_conf*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                st.markdown(f"""
                <div class="metric-box">
                    <h3>Fake Confidence</h3>
                    <h2 style="color: #ff4d4d;">{fake_conf*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)

def game():
    st.title("Detection Game")
    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_round" not in st.session_state:
        st.session_state.game_round = 1

    if st.session_state.game_round > 5:
        st.markdown(f"""
        <div class="metric-box">
            <h2>Game Over! Score: {st.session_state.game_score}/5</h2>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Play Again"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.rerun()
        return

    st.markdown(f"**Round {st.session_state.game_round} of 5** | Score: {st.session_state.game_score}")

    if "current_round" not in st.session_state:
        real_img = fetch_real_image()
        fake_img = fetch_fake_image()
        if real_img and fake_img:
            if random.random() < 0.5:
                st.session_state.current_round = {"left": real_img, "right": fake_img, "answer": "left"}
            else:
                st.session_state.current_round = {"left": fake_img, "right": real_img, "answer": "right"}

    if st.session_state.get("current_round"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.current_round["left"], use_container_width=True)
        with col2:
            st.image(st.session_state.current_round["right"], use_container_width=True)

        guess = st.radio("Which is real?", ["Left", "Right"], horizontal=True)
        if st.button("Submit Guess"):
            if guess.lower() == st.session_state.current_round["answer"]:
                st.session_state.game_score += 1
                st.success("Correct!")
            else:
                st.error("Wrong!")
            st.session_state.game_round += 1
            del st.session_state.current_round
            st.rerun()

# ----- Main App Flow -----
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    
    if st.session_state.page == "welcome":
        welcome()
    elif st.session_state.page == "game":
        game()
    else:
        main()
