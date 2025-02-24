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
    Make sure the folder 'game_real' exists and contains JPG images.
    """
    real_images = [os.path.join("game_real", f) for f in os.listdir("game_real") if f.lower().endswith(".jpg")]
    if real_images:
        return Image.open(random.choice(real_images))
    return None

def fetch_fake_image():
    """
    Fetch a fake image.
    Here we simply return a default fake image from the 'samples' directory.
    Adjust this function if you have a dedicated folder for fake images.
    """
    fake_image_path = "samples/fake_sample.jpg"
    if os.path.exists(fake_image_path):
        return Image.open(fake_image_path)
    return None
# ----- End of helper functions for fetching images -----

# Set page config before any other Streamlit commands
st.set_page_config(page_title="Deepfake Detective", page_icon="🕵️", layout="centered")

# Custom CSS for dark blue and cyan-blue theme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&display=swap');
        * { font-family: 'Space Grotesk', sans-serif; }
        .main { background: linear-gradient(135deg, #001f3f 0%, #00bcd4 100%); color: #ffffff; }
        .stButton>button { background: #00bcd4; color: white; border-radius: 15px; padding: 10px 24px; border: none; transition: all 0.3s ease; }
        .stButton>button:hover { background: #008ba3; transform: scale(1.05); }
        .stFileUploader>div>div>div>div { color: #ffffff; border: 2px dashed #00bcd4; background: rgba(0, 188, 212, 0.1); border-radius: 15px; }
        .metric-box { background: rgba(0, 188, 212, 0.1); padding: 20px; border-radius: 15px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .game-image { border: 3px solid transparent; border-radius: 15px; transition: all 0.3s ease; }
        .game-image:hover { transform: scale(1.02); cursor: pointer; }
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
# Main Page: Image Analysis
# =======================
def main():
    # Display logo using st.image() for reliability
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=300)
    except Exception as e:
        st.write("Logo not found.")

    st.title("Deepfake Detection System")
    
    with st.sidebar:
        st.markdown("""
        <div style="border-left: 3px solid #00bcd4; padding-left: 1rem; margin: 1rem 0;">
            <h2 style="color: #00bcd4;">🔍 Navigation</h2>
            <p>Test your skills in our detection challenge!</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎮 Start Detection Game", use_container_width=True):
            st.session_state.page = "game"
    
    col1, col2 = st.columns([4, 3])
    with col1:
        st.markdown("### 📤 Image Analysis Zone")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        sample_option = st.selectbox("Or choose from samples:", ["Select", "Real Sample", "Fake Sample"],
                                     help="Explore pre-loaded examples to test the system")
    with col2:
        if uploaded_file or sample_option != "Select":
            st.markdown("### 🔍 Preview")
            if uploaded_file:
                image = Image.open(uploaded_file)
            elif sample_option == "Real Sample":
                image = Image.open("samples/real_sample.jpg")
            else:
                image = Image.open("samples/fake_sample.jpg")
            st.image(image, use_container_width=True, caption="Selected Image Preview")
    
    if uploaded_file or sample_option != "Select":
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

    if st.session_state.game_round > 5:
        st.write(f"**Game Over! Your score: {st.session_state.game_score}/5**")
        if st.button("Play Again"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.session_state.used_real_images = set()
            st.session_state.pop("current_round_data", None)
            st.session_state.pop("round_submitted", None)
            st.session_state.pop("round_result", None)
        return

    st.write(f"Round {st.session_state.game_round} of 5")

    if "current_round_data" not in st.session_state:
        real_image = fetch_real_image()
        fake_image = fetch_fake_image()
        if real_image is None or fake_image is None:
            st.error("Failed to load images. Please try again.")
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

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            st.session_state.current_round_data["left_image"],
            caption="Left Image",
            use_container_width=True,
            key=f"left_{st.session_state.game_round}"
        )
    with col2:
        st.image(
            st.session_state.current_round_data["right_image"],
            caption="Right Image",
            use_container_width=True,
            key=f"right_{st.session_state.game_round}"
        )

    if not st.session_state.round_submitted:
        control_container = st.empty()
        with control_container.container():
            user_choice = st.radio("Which image is real?", ["Left", "Right"])
            if st.button("Submit", key="submit"):
                if user_choice == st.session_state.current_round_data["correct_answer"]:
                    st.session_state.game_score += 1
                    st.session_state.round_result = "Correct! 🎉"
                else:
                    st.session_state.round_result = "Wrong! 😢"
                st.session_state.round_submitted = True
                control_container.empty()

    if st.session_state.round_submitted:
        if st.session_state.round_result == "Correct! 🎉":
            st.success(st.session_state.round_result)
        else:
            st.error(st.session_state.round_result)
        if st.button("Next Round"):
            st.session_state.game_round += 1
            st.session_state.pop("current_round_data", None)
            st.session_state.pop("round_submitted", None)
            st.session_state.pop("round_result", None)

    if st.button("Go to Home"):
        st.session_state.page = "main"

# =======================
# Page Routing
# =======================
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "main"
    if st.session_state.page == "game":
        game()
    else:
        main()
