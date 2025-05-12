import streamlit as st
from PIL import Image
import io
import hashlib
import random
import os
import keras
import numpy as np
from keras import applications
import requests

# Set Keras backend to JAX
os.environ["KERAS_BACKEND"] = "jax"

# ----- Constants -----
API_USER = "1285106646"
API_KEY = "CDWtk3q6HdqHcs6DJxn9Y8YnL46kz6pX"

# ----- Image Fetching Functions -----
def fetch_real_image():
    real_dir = "game_real"
    if not os.path.exists(real_dir):
        st.error("Missing 'game_real' directory")
        return None
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not real_images:
        st.error("No real images found")
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
        st.error(f"Error loading image: {str(e)}")
        return None

def fetch_fake_image():
    fake_dir = "Game_Fake"
    if not os.path.exists(fake_dir):
        st.error("Missing 'Game_Fake' directory")
        return None
    fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not fake_images:
        st.error("No fake images found")
        return None
    if "used_fake_images" not in st.session_state:
        st.session_state.used_fake_images = set()
    available_images = [img for img in fake_images if img not in st.session_state.used_fake_images]
    if not available_images:
        st.session_state.used_fake_images = set()
        available_images = fake_images
    selected_image = random.choice(available_images)
    st.session_state.used_fake_images.add(selected_image)
    try:
        return Image.open(selected_image).copy()
    except Exception as e:
        st.error(f"Error loading fake image: {str(e)}")
        return None

# ----- AI Detection Functions -----
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model("deepfake_detection_model.h5", compile=False)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def get_image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()

@st.cache_data(show_spinner=False)
def predict_image(image_hash: str, _image: Image.Image):
    model = load_model()
    if model is None:
        return [{"label": "error", "score": 1.0}]
    img = _image.convert('RGB').resize((224, 224))
    img_array = applications.efficientnet.preprocess_input(np.array(img))
    try:
        prob = model.predict(np.expand_dims(img_array, axis=0))[0][0]
        return [
            {"label": "real", "score": float(prob)},
            {"label": "fake", "score": float(1 - prob)}
        ] if prob > 0.5 else [
            {"label": "fake", "score": float(1 - prob)},
            {"label": "real", "score": float(prob)}
        ]
    except Exception as e:
        return [{"label": "error", "score": 1.0}]

# ----- Sightengine API Integration -----
def analyze_with_sightengine(image_bytes):
    try:
        response = requests.post(
            'https://api.sightengine.com/1.0/check.json',
            files={'media': ('image.jpg', image_bytes, 'image/jpeg')},
            data={
                'api_user': API_USER,
                'api_secret': API_KEY,
                'models': 'deepfake,genai'
            }
        )
        result = response.json()
        scores = {
            'deepfake': result['type'].get('deepfake', 0.0) * 100,
            'ai_generated': result['type'].get('ai_generated', 0.0) * 100
        }
        return scores
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# ----- UI Components -----
def setup_page():
    st.set_page_config(page_title="DeepShield", page_icon="🕵️", layout="centered")
    st.markdown("""
    <style>
        :root {
            --primary: #00bcd4;
            --secondary: #001f3f;
            --accent: #ff4d4d;
            --success: #00ff88;
        }
        
        .main { 
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
            color: white;
        }
        
        .stButton>button {
            background: var(--primary);
            border: 2px solid white;
            border-radius: 25px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            background: var(--secondary);
            transform: scale(1.05);
        }
        
        .analysis-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            margin: 1rem 0;
        }
        
        .game-image-container {
            width: 100%;
            height: 300px;
            border-radius: 20px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        @media (max-width: 768px) {
            .game-image-container {
                height: 200px;
            }
            .stButton>button {
                padding: 0.5rem;
                font-size: 14px;
            }
        }
        
        .result-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .score-board {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem;
            background: rgba(0, 188, 212, 0.2);
            border-radius: 15px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

def welcome_page():
    try:
        st.image(Image.open("logo.png"), use_column_width=True)
    except:
        pass
    st.title("DeepShield AI Detector")
    st.markdown("""
    <div class="analysis-card">
        <h3>🕵️ Detect Deepfakes & AI-Generated Content</h3>
        <p>Advanced detection using state-of-the-art AI models</p>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div class="analysis-card">
                <h4>📸 Image Analysis</h4>
                <p>Dual detection systems</p>
            </div>
            <div class="analysis-card">
                <h4>🔐 Secure Processing</h4>
                <p>Military-grade encryption</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start Detection →"):
        st.session_state.page = "main"
        st.rerun()

# ----- Main Detection Interface -----
def main_interface():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        detection_mode = st.radio("Detection Mode", ["API Analysis", "Local Model"])
        st.markdown("---")
        st.markdown("## 🎮 Game Controls")
        if st.button("New Detection Game"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.session_state.page = "game"
            st.rerun()
        if st.button("← Return to Welcome"):
            st.session_state.page = "welcome"
            st.rerun()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("## 📤 Image Analysis")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        sample_option = st.selectbox("Or choose sample:", ["Select", "Real Sample", "Fake Sample"])

    with col2:
        if uploaded_file or sample_option != "Select":
            try:
                image = Image.open(uploaded_file) if uploaded_file else (
                    Image.open("samples/real_sample.jpg") if sample_option == "Real Sample" 
                    else Image.open("samples/fake_sample.jpg"))
                st.image(image, use_container_width=True)
            except Exception as e:
                st.error(f"Image Error: {str(e)}")

    if (uploaded_file or sample_option != "Select") and 'image' in locals():
        try:
            if detection_mode == "API Analysis":
                with st.spinner("🔍 Analyzing with Professional API..."):
                    image_bytes = uploaded_file.getvalue() if uploaded_file else (
                        open("samples/real_sample.jpg" if sample_option == "Real Sample" 
                            else "samples/fake_sample.jpg", "rb").read())
                    api_results = analyze_with_sightengine(image_bytes)
                    
                if api_results:
                    st.markdown("## 🔬 Analysis Report")
                    conclusion = (
                        "❌ Deepfake Detected" if api_results['deepfake'] > 40 else
                        "🤖 AI-Generated" if api_results['ai_generated'] > 40 else
                        "✅ Authentic Content"
                    )
                    
                    st.markdown(f"""
                    <div class="analysis-card">
                        <div style="text-align: center;">
                            <div class="result-icon">
                                {"❌" if '❌' in conclusion else "🤖" if '🤖' in conclusion else "✅"}
                            </div>
                            <h2>{conclusion}</h2>
                            <div class="score-board">
                                <div>
                                    <h4>Deepfake</h4>
                                    <h3>{api_results['deepfake']:.1f}%</h3>
                                </div>
                                <div>
                                    <h4>AI Generated</h4>
                                    <h3>{api_results['ai_generated']:.1f}%</h3>
                                </div>
                            </div>
                            <div class="confidence-bar">
                                <div style="width: {max(api_results['deepfake'], api_results['ai_generated'])}%; 
                                    height: 100%; 
                                    background: {'var(--accent)' if '❌' in conclusion or '🤖' in conclusion else 'var(--success)'};
                                    transition: width 0.5s ease;">
                                </div>
                            </div>
                            <p>Overall Confidence: {max(api_results['deepfake'], api_results['ai_generated']):.1f}%</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                with st.spinner("🤖 Analyzing with Local Model..."):
                    image_hash = get_image_hash(image)
                    model_results = predict_image(image_hash, image)
                    scores = {r["label"]: r["score"] for r in model_results}
                    
                st.markdown("## 📊 Local Analysis")
                st.markdown(f"""
                <div class="analysis-card">
                    <div style="text-align: center;">
                        <div class="result-icon">
                            {"❌" if scores['fake'] > 0.5 else "✅"}
                        </div>
                        <h2>{'❌ Fake Content Detected' if scores['fake'] > 0.5 else '✅ Authentic Content'}</h2>
                        <div class="score-board">
                            <div>
                                <h4>Real Confidence</h4>
                                <h3>{scores['real']*100:.1f}%</h3>
                            </div>
                            <div>
                                <h4>Fake Confidence</h4>
                                <h3>{scores['fake']*100:.1f}%</h3>
                            </div>
                        </div>
                        <div class="confidence-bar">
                            <div style="width: {max(scores['real'], scores['fake'])*100}%; 
                                height: 100%; 
                                background: {'var(--accent)' if scores['fake'] > 0.5 else 'var(--success)'};
                                transition: width 0.5s ease;">
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

# ----- Game Interface -----
def game_interface():
    st.title("🎮 Detection Training")
    
    with st.sidebar:
        if st.button("← Return to Main"):
            st.session_state.page = "main"
            st.rerun()

    if st.session_state.get("game_round", 1) > 5:
        st.markdown(f"""
        <div class="analysis-card" style="text-align: center;">
            <h2>Game Over! 🎯</h2>
            <h1 style="color: var(--primary);">{st.session_state.game_score}/5</h1>
            <div style="display: grid; gap: 1rem; margin-top: 2rem;">
                <button class="stButton" onclick="window.location.reload()">Play Again 🔄</button>
                <button class="stButton" onclick="window.location.href='/'">Return Home 🏠</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="score-board">
        <span>Round {st.session_state.game_round}/5</span>
        <span>Score: {st.session_state.game_score}</span>
    </div>
    """, unsafe_allow_html=True)

    if "current_round" not in st.session_state:
        real_img = fetch_real_image()
        fake_img = fetch_fake_image()
        if real_img and fake_img:
            st.session_state.current_round = {
                "images": random.sample([(real_img, "Real"), (fake_img, "Fake")], 2),
                "answer": random.choice(["Left", "Right"]),
                "images_base64": [
                    image_to_base64(real_img.resize((400, 400))),
                    image_to_base64(fake_img.resize((400, 400)))
                ]
            }

    if "current_round" in st.session_state:
        cols = st.columns(2)
        for idx in range(2):
            with cols[idx]:
                st.markdown(f"""
                <div class="game-image-container">
                    <img src="data:image/png;base64,{st.session_state.current_round['images_base64'][idx]}" 
                        style="width: 100%; height: 100%; object-fit: cover;">
                </div>
                """, unsafe_allow_html=True)

        user_guess = st.radio("Which image is real?", ["Left", "Right"], horizontal=True)
        
        if st.button("Submit Answer"):
            if user_guess == st.session_state.current_round["answer"]:
                st.session_state.game_score += 1
                st.success("Correct! 🎉")
            else:
                st.error("Wrong Answer 😢")
            st.session_state.game_round += 1
            del st.session_state.current_round
            st.rerun()

# ----- App Flow -----
if __name__ == "__main__":
    setup_page()
    if "page" not in st.session_state:
        st.session_state.page = "welcome"
    
    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "game":
        game_interface()
    else:
        main_interface()
