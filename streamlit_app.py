# Load model directly
from transformers import AutoProcessor, AutoModelForImageClassification
import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
import io
import hashlib
import random
import os
from huggingface_hub import hf_hub_download
import keras
import numpy as np

# ----- Human Verification System -----
@st.cache_resource
def load_human_model():
    processor = AutoProcessor.from_pretrained("prithivMLmods/Human-vs-NonHuman-Detection")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Human-vs-NonHuman-Detection")
    return processor, model

def is_human(image: Image.Image) -> bool:
    """Check if image contains a human face"""
    try:
        processor, model = load_human_model()
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        return model.config.id2label[predicted_class] == 'human'
    except Exception as e:
        st.error(f"Human verification error: {str(e)}")
        return False

# ----- Helper functions for fetching images for the game -----
def fetch_real_image():
    """Fetch a random real image from the 'game_real' directory."""
    real_dir = "game_real"
    if not os.path.exists(real_dir):
        st.error("Error: 'game_real' directory not found.")
        return None

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not real_images:
        st.error("Error: No images in 'game_real' directory.")
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
    """Fetch a fake image from the 'Game_Fake' directory."""
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

# ----- Helper functions -----
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ----- App Configuration -----
st.set_page_config(page_title="Deepfake", page_icon="🕵️", layout="centered")

# Custom CSS
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
    </style>""", unsafe_allow_html=True)

# ----- Deepfake Model -----
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="musabalosimi/deepfake1", filename="my_model1.keras")
    model = keras.models.load_model(model_path)
    return model

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
    
    if prob > 0.5:
        return [{"label": "real", "score": float(prob)}, {"label": "fake", "score": float(1 - prob)}]
    else:
        return [{"label": "fake", "score": float(1 - prob)}, {"label": "real", "score": float(prob)}]

# ----- Page Components -----
def welcome():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=800)
    except Exception as e:
        st.write("")

    st.title("DeepShield")

    st.markdown("""<div class="welcome-section">
        <h3>Welcome to Deepfake Detection</h3>
        <p>Detect AI-generated images and test your skills!</p>
        <h3>🕵️ Mady By </h3>
        <li>Musab Alosaimi</li>
        <li>Bassam Alanazi</li>
        <li>Abdulazlz AlHwitan</li>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="welcome-section">
            <h3>🎯 Features</h3>
            <ul>
                <li>Image Analysis</li>
                <li>Detection Game</li>
                <li>Real-time Predictions</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="welcome-section">
            <h3>🕹️ How to Use</h3>
            <ol>
                <li>Upload an image</li>
                <li>Check results</li>
                <li>Play detection game</li>
            </ol>
        </div>""", unsafe_allow_html=True)

    if st.button("Get Started →", use_container_width=True):
        st.session_state.page = "main"
        rerun()

def main():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=800)
    except Exception as e:
        st.write("")

    st.title("Deepfake Detection System")

    with st.sidebar:
        st.markdown("""<div style="border-left: 3px solid #00bcd4; padding-left: 1rem;">
            <h2 style="color: #00bcd4;">🔍 Navigation</h2>
        </div>""", unsafe_allow_html=True)
        if st.button("🏠 Return to Welcome", use_container_width=True):
            st.session_state.page = "welcome"
            rerun()
        if st.button("🎮 Start Detection Game", use_container_width=True):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.session_state.page = "game"
            rerun()

    col1, col2 = st.columns([4, 3])
    with col1:
        st.markdown("### 📤 Image Analysis Zone")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        sample_option = st.selectbox(
            "Or choose samples:",
            ["Select", "Real Sample", "Fake Sample"],
            help="Pre-loaded examples"
        )

    with col2:
        if uploaded_file or sample_option != "Select":
            try:
                image = None
                if uploaded_file:
                    image = Image.open(uploaded_file)
                elif sample_option == "Real Sample":
                    image = Image.open("samples/real_sample.jpg") if os.path.exists("samples/real_sample.jpg") else None
                elif sample_option == "Fake Sample":
                    image = Image.open("samples/fake_sample.jpg") if os.path.exists("samples/fake_sample.jpg") else None

                if image:
                    st.image(image, use_container_width=True, caption="Preview")
                    # Human verification
                    if not is_human(image):
                        st.error("🚫 No human detected! Please upload a human image.")
                        return
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if (uploaded_file or sample_option != "Select") and 'image' in locals() and image is not None:
        try:
            with st.spinner("🔬 Analyzing..."):
                image_hash = get_image_hash(image)
                result = predict_image(image_hash, image)
                scores = {r["label"].lower(): r["score"] for r in result}

            st.markdown("---")
            st.markdown("### 📊 Report")

            col1, col2 = st.columns(2)
            with col1:
                real_conf = scores.get("real", 0)
                real_chart = alt.Chart(pd.DataFrame({"Category": ["Real"], "Confidence": [real_conf]})).mark_bar().encode(
                    x='Category', y='Confidence')
                st.altair_chart(real_chart, use_container_width=True)

            with col2:
                fake_conf = scores.get("fake", 0)
                fake_chart = alt.Chart(pd.DataFrame({"Category": ["Fake"], "Confidence": [fake_conf]})).mark_bar().encode(
                    x='Category', y='Confidence')
                st.altair_chart(fake_chart, use_container_width=True)

            final_pred = max(scores, key=scores.get)
            if final_pred == "fake":
                st.markdown(f"""
                <div style="background: rgba(255,77,77,0.2); padding: 1rem; border-radius: 15px;">
                    <h3>🚨 AI Detected! ({scores[final_pred]*100:.2f}%)</h3>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(0,255,136,0.2); padding: 1rem; border-radius: 15px;">
                    <h3>✅ Authentic ({scores[final_pred]*100:.2f}%)</h3>
                </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

def game():
    st.title("Deepfake Game")
    st.write("Guess the real image (5 rounds)")

    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_round" not in st.session_state:
        st.session_state.game_round = 1

    if st.session_state.game_round > 5:
        st.markdown(f"""
        <div style="background: rgba(0,188,212,0.2); padding: 2rem; border-radius: 15px;">
            <h2>🎮 Game Over! Score: {st.session_state.game_score}/5</h2>
        </div>""", unsafe_allow_html=True)

        if st.button("Play Again"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            rerun()
        return

    st.markdown(f"""
    <div style="background: rgba(0,188,212,0.1); padding: 1rem; border-radius: 15px;">
        <h3>Round {st.session_state.game_round} | Score: {st.session_state.game_score}</h3>
    </div>""", unsafe_allow_html=True)

    real_image = fetch_real_image()
    fake_image = fetch_fake_image()

    if real_image and fake_image:
        left_image, right_image = (real_image, fake_image) if random.choice([True, False]) else (fake_image, real_image)
        correct = "Left" if left_image == real_image else "Right"

        col1, col2 = st.columns(2)
        with col1:
            st.image(left_image.resize((300, 300)), caption="Left")
        with col2:
            st.image(right_image.resize((300, 300)), caption="Right")

        choice = st.radio("Choose:", ["Left", "Right"], horizontal=True)
        if st.button("Submit"):
            if choice == correct:
                st.session_state.game_score += 1
                st.success("Correct! 🎉")
            else:
                st.error("Wrong! 😢")
            
            if st.button("Next Round"):
                st.session_state.game_round += 1
                rerun()

# ----- App Flow -----
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        welcome()
    elif st.session_state.page == "game":
        game()
    else:
        main()
