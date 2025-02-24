import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import altair as alt
import io
import hashlib
import random
import os
import time
from datetime import datetime

# ----- Configuration -----
MODERN_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .main { 
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        border: none;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
    }
    
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea33 0%, #764ba233 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .game-card {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .game-card:hover {
        transform: scale(1.02);
    }
    
    .progress-bar {
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.5s ease;
    }
</style>
"""

# ----- Helper Functions -----
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

def get_image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()

@st.cache_data(show_spinner=False)
def predict_image(image_hash: str, _image: Image.Image):
    model = load_model()
    return model(_image)

# ----- Advanced Components -----
def create_progress_bar(current, total):
    progress = current / total
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {progress * 100}%"></div>
    </div>
    """

def create_confidence_gauge(confidence):
    color = "#00ff88" if confidence < 0.5 else "#ff4d4d"
    return alt.Chart(pd.DataFrame({"value": [confidence]})).mark_arc().encode(
        theta=alt.Theta(field="value", type="quantitative"),
        color=alt.ColorValue(color),
        tooltip=[alt.Tooltip('value', title='Confidence')]
    ).properties(width=150, height=150)

# ----- Pages -----
def main_page():
    with st.container():
        st.markdown(f'<div class="glass-container"><h1 style="margin-bottom:0;">🔍 DeepVision Analyzer</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("""
            ### Next-Gen Deepfake Detection
            Upload an image or try our samples to analyze potential AI manipulations.
            Our advanced neural network provides detailed authenticity insights.
            """)
            
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            sample_option = st.selectbox("Or try a sample:", ["Select", "Real Sample", "Fake Sample"])
            
            if uploaded_file or sample_option != "Select":
                if uploaded_file:
                    image = Image.open(uploaded_file)
                else:
                    sample_path = "samples/real_sample.jpg" if sample_option == "Real Sample" else "samples/fake_sample.jpg"
                    image = Image.open(sample_path) if os.path.exists(sample_path) else None
                
                if image:
                    st.image(image, use_container_width=True)
                    
                    with st.spinner("🧠 Analyzing image patterns..."):
                        image_hash = get_image_hash(image)
                        result = predict_image(image_hash, image)
                        scores = {r["label"].lower(): r["score"] for r in result}
                        
                    st.markdown("### Analysis Report")
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Authenticity Score</h3>
                            <h1 style="color: #00ff88; margin:0;">{scores.get('real', 0)*100:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_metric2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>AI Probability</h3>
                            <h1 style="color: #ff4d4d; margin:0;">{scores.get('fake', 0)*100:.1f}%</h1>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.altair_chart(create_confidence_gauge(scores.get('fake', 0)), use_container_width=True)
                    
                    if scores['fake'] > 0.7:
                        st.markdown("""
                        <div style="border-left: 4px solid #ff4d4d; padding-left: 1rem; margin: 2rem 0;">
                            <h3>🚨 High AI Detection Alert</h3>
                            <p>This image shows strong signs of artificial generation. Look for these indicators:</p>
                            <ul>
                                <li>Unnatural skin textures</li>
                                <li>Inconsistent lighting/shadow</li>
                                <li>Asymmetrical facial features</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-container" style="margin-top: 4rem;">
                <h3>📈 Recent Analysis History</h3>
                <div style="margin: 1rem 0;">
                    <div style="display: flex; justify-content: space-between; margin: 0.5rem 0;">
                        <span>2023-11-15 14:30</span>
                        <span style="color: #00ff88">85% Real</span>
                    </div>
                    """ + create_progress_bar(3, 5) + """
                </div>
                <div style="text-align: center; margin-top: 2rem;">
                    <button style="background: transparent; border: 1px solid #667eea; color: #667eea; padding: 8px 20px; border-radius: 8px;">
                        View Full History
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)

def game_page():
    st.markdown(f'<div class="glass-container"><h1 style="margin-bottom:0;">🎮 Deepfake Detective Challenge</h1>', unsafe_allow_html=True)
    
    if "game_state" not in st.session_state:
        st.session_state.game_state = {
            "score": 0,
            "round": 1,
            "streak": 0,
            "best_streak": 0
        }
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; margin: 2rem 0;">
        <div class="metric-card">
            <h4>Current Round</h4>
            <h1>{st.session_state.game_state['round']}/5</h1>
        </div>
        <div class="metric-card">
            <h4>Score</h4>
            <h1>{st.session_state.game_state['score']}</h1>
        </div>
        <div class="metric-card">
            <h4>Best Streak</h4>
            <h1>{st.session_state.game_state['best_streak']}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Game logic here...
    
# ----- App Setup -----
st.markdown(MODERN_CSS, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "main"

with st.sidebar:
    st.markdown("""
    <div class="glass-container" style="padding: 1.5rem;">
        <h2>Navigation</h2>
        <div style="margin: 1rem 0;">
            <button onclick="window.streamlit.setComponentValue('main')" style="width: 100%; margin: 0.5rem 0;">
                🕵️ Analysis Tool
            </button>
            <button onclick="window.streamlit.setComponentValue('game')" style="width: 100%; margin: 0.5rem 0;">
                🎮 Detection Game
            </button>
            <button style="width: 100%; margin: 0.5rem 0;">
                📚 Education Hub
            </button>
            <button style="width: 100%; margin: 0.5rem 0;">
                📊 User Stats
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "game":
    game_page()
