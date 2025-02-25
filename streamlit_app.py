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

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
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

    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
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

# Helper function for page reruns
def rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# Set page config before any other Streamlit commands
st.set_page_config(page_title="Deepfake", page_icon="🕵️", layout="centered")

# Custom CSS for dark blue and cyan-blue theme
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
    </style>
""", unsafe_allow_html=True)

# Load the deepfake detection model (cached so it loads only once)
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

# Helper function to generate a hash for a PIL image
def get_image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()

# Cache predictions based on the image hash.
@st.cache_data(show_spinner=False)
def predict_image(image_hash: str, _image: Image.Image):
    model = load_model()
    return model(_image)

# =======================
# Welcome Page
# =======================
def welcome():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=800)
    except Exception as e:
        st.write("")

    st.title("    DeepShield    ")

    st.markdown("""
    <div class="welcome-section">
        <h3>Welcome to the Deepfake Detection final project</h3>
        <p>This application helps you detect AI-generated images using state-of-the-art machine learning models. 
        Explore the capabilities of deepfake detection through image analysis or test your skill in the detection challenge game!</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="welcome-section">
            <h3>🎯 Key Features</h3>
            <ul>
                <li><strong>Image Analysis</strong>: Upload an image to check if it's real or AI-generated</li>
                <li><strong>Detection Game</strong>: Train your eye to spot deepfakes in a fun, interactive game</li>
                <li><strong>Real-time Predictions</strong>: Get instant results with confidence scores</li>
                <li><strong>Educational Insights</strong>: Learn about deepfake technology and detection methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="welcome-section">
            <h3>🕹️ How to Use</h3>
            <ol>
                <li>Use the <strong>Image Analysis</strong> page to check individual images</li>
                <li>Try the <strong>Detection Game</strong> to test your detection skills</li>
                <li>Review the confidence metrics to understand model predictions</li>
                <li>Explore different sample images to see varied results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col3:
            st.markdown("""
            <div style="
                 background: rgba(0,188,212,0.15);
                 border: 2px solid #00bcd4;
                 padding: 20px;
                 border-radius: 0px;
                 margin: 10px;
                 text-align: center;
                 box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                 ">
                <h2 style="color: #00bcd4; margin-bottom: 10px;">🕵️ Made by</h2>
                <ul style="list-style-type: none; padding: 0; margin: 0;">
                    <li style="font-weight: bold; color: #ffffff;">Musab Alosaimi</li>
                    <li style="font-weight: bold; color: #ffffff;">Bassam Alanazi</li>
                    <li style="font-weight: bold; color: #ffffff;">Abdulazlz AlHwitan</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    if st.button("Get Started →", use_container_width=True):
        st.session_state.page = "main"
        rerun()

# =======================
# Main Page: Image Analysis
# =======================
def main():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=800)
    except Exception as e:
        st.write("")

    st.title("Deepfake Detection System")

    with st.sidebar:
        st.markdown("""
        <div style="border-left: 3px solid #00bcd4; padding-left: 1rem; margin: 1rem 0;">
            <h2 style="color: #00bcd4;">🔍 Navigation</h2>
            <p>Navigate between different sections</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🏠 Return to Welcome", use_container_width=True):
            st.session_state.page = "welcome"
            rerun()
        if st.button("🎮 Start Detection Game", use_container_width=True):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            if "used_real_images" in st.session_state:
                st.session_state.used_real_images = set()
            for key in ["current_round_data", "round_submitted", "round_result"]:
                if key in st.session_state:
                    st.session_state.pop(key)
            st.session_state.page = "game"
            rerun()

    col1, col2 = st.columns([4, 3])
    with col1:
        st.markdown("### 📤 Image Analysis Zone")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        sample_option = st.selectbox("Or choose from samples:", ["Select", "Real Sample", "Fake Sample"],
                                     help="Explore pre-loaded examples to test the system")
    with col2:
        if uploaded_file or sample_option != "Select":
            st.markdown("### 🔍 Preview")
            try:
                if uploaded_file:
                    image = Image.open(uploaded_file)
                elif sample_option == "Real Sample":
                    if os.path.exists("samples/real_sample.jpg"):
                        image = Image.open("samples/real_sample.jpg")
                    else:
                        st.error("Real sample image not found. Please check 'samples/real_sample.jpg'")
                        image = None
                else:
                    if os.path.exists("samples/fake_sample.jpg"):
                        image = Image.open("samples/fake_sample.jpg")
                    else:
                        st.error("Fake sample image not found. Please check 'samples/fake_sample.jpg'")
                        image = None

                if image:
                    st.image(image, use_container_width=True, caption="Selected Image Preview")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    if (uploaded_file or sample_option != "Select") and 'image' in locals() and image is not None:
        try:
            with st.spinner("🔬 Scanning image for AI fingerprints..."):
                image_hash = get_image_hash(image)
                result = predict_image(image_hash, image)
                scores = {r["label"].lower(): r["score"] for r in result}
            st.markdown("---")
            st.markdown("### 📊 Detection Report")

            col_chart_left, col_chart_right = st.columns(2)
            with col_chart_left:
                real_chart_data = pd.DataFrame({"Category": ["Real"], "Confidence": [scores.get("real", 0)]})
                real_chart = alt.Chart(real_chart_data).mark_bar(size=40, color="#00ff88").encode(
                    x=alt.X("Category", title=""),
                    y=alt.Y("Confidence", title="Confidence", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["Category", "Confidence"]
                ).properties(height=200)
                st.altair_chart(real_chart, use_container_width=True)
            with col_chart_right:
                fake_chart_data = pd.DataFrame({"Category": ["Fake"], "Confidence": [scores.get("fake", 0)]})
                fake_chart = alt.Chart(fake_chart_data).mark_bar(size=40, color="#ff4d4d").encode(
                    x=alt.X("Category", title=""),
                    y=alt.Y("Confidence", title="Confidence", scale=alt.Scale(domain=[0, 1])),
                    tooltip=["Category", "Confidence"]
                ).properties(height=200)
                st.altair_chart(fake_chart, use_container_width=True)

            final_pred = max(scores, key=scores.get)
            if final_pred == "fake":
                st.markdown(f"""
                <div style="background: rgba(255,77,77,0.2); padding: 1rem; border-radius: 15px; border-left: 5px solid #ff4d4d;">
                    <h3 style="margin:0;">🚨 AI Detected! ({scores[final_pred]*100:.2f}% confidence)</h3>
                    <p style="margin:0; opacity:0.8;">This image shows signs of artificial generation</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(0,255,136,0.2); padding: 1rem; border-radius: 15px; border-left: 5px solid #00ff88;">
                    <h3 style="margin:0;">✅ Authentic Content ({scores[final_pred]*100:.2f}% confidence)</h3>
                    <p style="margin:0; opacity:0.8;">No significant AI manipulation detected</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"🔧 Analysis error: {str(e)}")

# =======================
# Game Page: Swipe-based Detection Challenge
# =======================
def game():
    st.title("Deepfake Game")
    st.write("Guess which image is real! You have 5 rounds.")

    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_round" not in st.session_state:
        st.session_state.game_round = 1
    if "used_real_images" not in st.session_state:
        st.session_state.used_real_images = set()

    if st.session_state.game_round > 5:
        st.markdown(f"""
        <div style="background: rgba(0,188,212,0.2); padding: 2rem; border-radius: 15px; text-align: center;">
            <h2 style="margin-bottom:1rem;">🎮 Game Over!</h2>
            <h3>Your score: {st.session_state.game_score}/5</h3>
            <p>Can you do better next time?</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Play Again"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.session_state.used_real_images = set()
            for key in ["current_round_data", "round_submitted", "round_result"]:
                if key in st.session_state:
                    st.session_state.pop(key)
            rerun()

        if st.button("Return to Home"):
            st.session_state.page = "main"
            rerun()
        return

    st.markdown(f"""
    <div style="background: rgba(0,188,212,0.1); padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1.5rem;">
        <h3 style="margin:0;">Round {st.session_state.game_round} of 5</h3>
        <p style="margin:0;">Current Score: {st.session_state.game_score}</p>
    </div>
    """, unsafe_allow_html=True)

    if "current_round_data" not in st.session_state or not st.session_state.current_round_data:
        real_image = fetch_real_image()
        fake_image = fetch_fake_image()

        if real_image is None or fake_image is None:
            st.error("Could not load game images. Please check that your 'game_real' and 'Game_Fake' directories contain valid images.")
            if st.button("Return to Main Page"):
                st.session_state.page = "main"
                rerun()
            return

        if random.choice([True, False]):
            left_image, right_image = real_image, fake_image
            correct_answer = "Left"
        else:
            left_image, right_image = fake_image, real_image
            correct_answer = "Right"

        st.session_state.current_round_data = {
            "left_image": left_image,
            "right_image": right_image,
            "correct_answer": correct_answer,
        }
        st.session_state.round_submitted = False
        st.session_state.round_result = None

    fixed_size = (300, 300)
    left_fixed = st.session_state.current_round_data["left_image"].resize(fixed_size)
    right_fixed = st.session_state.current_round_data["right_image"].resize(fixed_size)
    col1, col2 = st.columns(2)
    with col1:
        st.image(left_fixed, caption="Left Image", use_container_width=False)
    with col2:
        st.image(right_fixed, caption="Right Image", use_container_width=False)

    if "round_submitted" not in st.session_state or not st.session_state.round_submitted:
        st.markdown("### Which image do you think is real?")
        user_choice = st.radio("Select one:", ["Left", "Right"], horizontal=True, label_visibility="collapsed")

        col_button1, col_button2 = st.columns([1, 1])
        with col_button1:
            if st.button("Submit Answer", use_container_width=True):
                correct = user_choice == st.session_state.current_round_data["correct_answer"]
                if correct:
                    st.session_state.game_score += 1
                    st.session_state.round_result = "Correct! 🎉"
                else:
                    st.session_state.round_result = "Wrong! 😢"
                st.session_state.round_submitted = True
                rerun()

    if st.session_state.get("round_submitted", False):
        if st.session_state.round_result == "Correct! 🎉":
            st.success(f"{st.session_state.round_result} The {st.session_state.current_round_data['correct_answer']} image is real.")
        else:
            st.error(f"{st.session_state.round_result} The {st.session_state.current_round_data['correct_answer']} image is real.")

        if st.button("Next Round", use_container_width=True):
            st.session_state.game_round += 1
            for key in ["current_round_data", "round_submitted", "round_result"]:
                if key in st.session_state:
                    st.session_state.pop(key)
            rerun()

    st.markdown("---")
    if st.button("Exit Game", use_container_width=True):
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
    elif st.session_state.page == "game":
        game()
    else:
        main()
