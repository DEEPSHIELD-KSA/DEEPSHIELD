import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import altair as alt
import requests
import os
import random

# Set page config before any other Streamlit commands
st.set_page_config(page_title="Deepfake Detective", page_icon="🕵️", layout="centered")

# Custom CSS for styling
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
        background: #e94560;
        color: white;
        border-radius: 15px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #ff6b6b;
        transform: scale(1.05);
    }
    
    .stFileUploader>div>div>div>div {
        color: #ffffff;
        border: 2px dashed #e94560;
        background: rgba(233, 69, 96, 0.1);
        border-radius: 15px;
    }
    
    .metric-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .game-image {
        border: 3px solid transparent;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .game-image:hover {
        transform: scale(1.02);
        cursor: pointer;
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
    # Header Section
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 2.5rem; margin: 0; color: #e94560; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🕵️ Deepfake Detective
            </h1>
            <p style="font-size: 1.1rem; opacity: 0.9;">
                Unmask AI-generated images with cutting-edge detection technology
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with Navigation
    with st.sidebar:
        st.markdown("""
            <div style="border-left: 3px solid #e94560; padding-left: 1rem; margin: 1rem 0;">
                <h2 style="color: #e94560;">🔍 Navigation</h2>
                <p>Test your skills in our detection challenge!</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🎮 Start Detection Game", use_container_width=True):
            st.session_state.page = "game"
            st.experimental_rerun()

    # Main Content: Upload or choose sample image
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 📤 Image Analysis Zone")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        sample_option = st.selectbox("Or choose from samples:", 
                                     ["Select", "Real Sample", "Fake Sample"],
                                     help="Explore pre-loaded examples to test the system")
    with col2:
        if uploaded_file or sample_option != "Select":
            st.markdown("### 🔍 Preview")
            preview_image = (Image.open(uploaded_file) if uploaded_file 
                             else (Image.open("samples/real_sample.jpg") if sample_option == "Real Sample" 
                                   else Image.open("samples/fake_sample.jpg")))
            st.image(preview_image, use_container_width=True, caption="Selected Image Preview")

    # Analysis Section
    if uploaded_file or sample_option != "Select":
        try:
            with st.spinner("🔬 Scanning image for AI fingerprints..."):
                model = load_model()
                result = model(preview_image)
                scores = {r["label"].lower(): r["score"] for r in result}
            st.markdown("---")
            st.markdown("### 📊 Detection Report")
            
            # Animated Metrics
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"""
                    <div class="metric-box">
                        <h3 style="margin:0; color: #00ff88">REAL</h3>
                        <h1 style="margin:0; font-size: 2.5rem">{scores.get('real', 0) * 100:.2f}%</h1>
                    </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""
                    <div class="metric-box">
                        <h3 style="margin:0; color: #ff4d4d">FAKE</h3>
                        <h1 style="margin:0; font-size: 2.5rem">{scores.get('fake', 0) * 100:.2f}%</h1>
                    </div>
                """, unsafe_allow_html=True)

            # Enhanced Chart using Altair
            chart_data = pd.DataFrame({
                "Category": ["Real", "Fake"],
                "Confidence": [scores.get("real", 0), scores.get("fake", 0)],
            })
            chart = alt.Chart(chart_data).mark_bar(size=40).encode(
                x=alt.X("Category", title="", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Confidence", title="Confidence", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Category", scale=alt.Scale(domain=["Real", "Fake"], 
                               range=["#00ff88", "#ff4d4d"]), legend=None),
                tooltip=["Category", "Confidence"]
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)

            # Result Badge
            final_pred = max(scores, key=scores.get)
            if final_pred == "fake":
                st.markdown(f"""
                    <div style="background: #ff4d4d33; padding: 1rem; border-radius: 15px; border-left: 5px solid #ff4d4d;">
                        <h3 style="margin:0;">🚨 AI Detected! ({scores[final_pred]*100:.2f}% confidence)</h3>
                        <p style="margin:0; opacity:0.8;">This image shows signs of artificial generation</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background: #00ff8833; padding: 1rem; border-radius: 15px; border-left: 5px solid #00ff88;">
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
    # Initialize game state variables if not already set
    if 'game_round' not in st.session_state:
        st.session_state.game_round = 1
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'current_image' not in st.session_state:
        st.session_state.current_image = random.choice(['real', 'fake'])

    # End game after 5 rounds
    if st.session_state.game_round > 5:
        st.markdown(f"### Game Over! Your score: {st.session_state.score} / 5")
        if st.button("Play Again"):
            st.session_state.game_round = 1
            st.session_state.score = 0
            st.session_state.current_image = random.choice(['real', 'fake'])
            st.experimental_rerun()
        return

    # Game Header and Progress Indicator
    st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 2.5rem; margin: 0; color: #e94560;">🕹️ Detection Challenge</h1>
            <p style="font-size: 1.1rem; opacity: 0.9;">Spot the real one! Round {st.session_state.game_round}/5</p>
            <div style="background: #e94560; width: {(st.session_state.game_round - 1) / 5 * 100}%; height: 4px; margin: 0 auto;"></div>
        </div>
    """, unsafe_allow_html=True)

    # Display current image based on game state
    image_path = "samples/real_sample.jpg" if st.session_state.current_image == 'real' else "samples/fake_sample.jpg"
    image = Image.open(image_path)
    st.image(image, caption="Swipe left for REAL, right for FAKE", use_container_width=True)

    # Buttons for swiping: Left = Real, Right = Fake
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Swipe Left (Real)"):
            guess = 'real'
            if guess == st.session_state.current_image:
                st.success("Correct!")
                st.session_state.score += 1
            else:
                st.error("Incorrect!")
            st.session_state.game_round += 1
            st.session_state.current_image = random.choice(['real', 'fake'])
            st.experimental_rerun()
    with col2:
        if st.button("Swipe Right (Fake)"):
            guess = 'fake'
            if guess == st.session_state.current_image:
                st.success("Correct!")
                st.session_state.score += 1
            else:
                st.error("Incorrect!")
            st.session_state.game_round += 1
            st.session_state.current_image = random.choice(['real', 'fake'])
            st.experimental_rerun()

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
