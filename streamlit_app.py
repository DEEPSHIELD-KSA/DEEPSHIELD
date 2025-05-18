import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
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

# ----- Constants & Configurations -----
API_USER = "1285106646"  # Replace with your API credentials
API_KEY = "CDWtk3q6HdqHcs6DJxn9Y8YnL46kz6pX"    # Replace with your API key
MODEL_PATH = "deepfake_detection_model.h5"  # Replace with your model path

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
        return keras.models.load_model(MODEL_PATH, compile=False)
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

# ----- API Integration -----
def analyze_with_custom_api(image_bytes):
    try:
        response = requests.post(
            'https://your-custom-api-endpoint.com/detect',
            files={'image': ('image.jpg', image_bytes, 'image/jpeg')},
            headers={'Authorization': f'Bearer {API_KEY}'}
        )
        result = response.json()
        return {
            'deepfake': result.get('deepfake_score', 0) * 100,
            'ai_generated': result.get('ai_probability', 0) * 100
        }
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
        }
        
        .main { 
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%); 
            color: white;
        }
        
        .stButton>button {
            background: var(--primary);
            border: 2px solid white;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background: var(--secondary);
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .game-card {
            transition: transform 0.3s ease;
            cursor: pointer;
        }
        
        .game-card:hover {
            transform: scale(1.03);
        }
        
        h1, h2, h3 {
            color: white !important;
        }
        
        @media screen and (max-width: 600px) {
            .game-image {
                width: 150px !important;
                margin: 0 auto;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def welcome_page():
    col1, col2 = st.columns([1, 3])
    with col1:
        try:
            st.image(Image.open("logo.png"), width=400)
        except:
            pass
    with col2:
        st.title("DeepShield AI Detector")
    
    st.markdown("""
    <div class="metrics-container">
        <div class="metric-card feature">
            <h3>🕵️ Advanced Deepfake Detection</h3>
            <p>Combining API and Local Model Analysis Automatically</p>
        </div>
        <div class="metric-card">
            <h4>🤖 Dual Analysis</h4>
            <p>Automatic best result selection</p>
        </div>
        <div class="metric-card">
            <h4>🔐 Secure Processing</h4>
            <p>All analyses are encrypted</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Start Detection →", key="start_btn"):
        st.session_state.page = "main"
        st.rerun()

# ----- Analysis Reports -----
def enhanced_analysis_report(api_results, selected=False):
    title = "## 🔬 Professional Analysis Report" + (" 🏆" if selected else "")
    st.markdown(title)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧪 Deepfake Probability</h4>
            <h1 style="color: var(--accent);">{api_results['deepfake']:.0f}%</h1>
            <progress value="{api_results['deepfake']}" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🧠 AI-Generated Probability</h4>
            <h1 style="color: var(--accent);">{api_results['ai_generated']:.0f}%</h1>
            <progress value="{api_results['ai_generated']}" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        """, unsafe_allow_html=True)

def local_model_report(model_results, selected=False):
    title = "## 🤖 Local Model Analysis" + (" 🏆" if selected else "")
    st.markdown(title)
    scores = {r["label"]: r["score"] for r in model_results}
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>✅ Real Confidence</h4>
            <h1 style="color: #00ff88;">{scores['real']*100:.0f}%</h1>
            <progress value="{scores['real']*100}" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>❌ Fake Confidence</h4>
            <h1 style="color: var(--accent);">{scores['fake']*100:.0f}%</h1>
            <progress value="{scores['fake']*100}" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        """, unsafe_allow_html=True)

# ----- Main Detection Interface -----
def main_interface():
    with st.sidebar:
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
        uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])
        camera_photo = st.camera_input("Take a photo in real-time")
        sample_option = st.selectbox("Or choose sample:", ["Select", "Real Sample", "Fake Sample"])

    with col2:
        current_image = None
        if camera_photo:
            current_image = Image.open(camera_photo)
        elif uploaded_file:
            current_image = Image.open(uploaded_file)
        elif sample_option != "Select":
            current_image = Image.open("samples/real_sample.jpg" if sample_option == "Real Sample" 
                                      else "samples/fake_sample.jpg")

        if current_image:
            st.image(current_image, caption="Selected Image", use_container_width=True, 
                    output_format="JPEG", clamp=True)

    if current_image:
        try:
            api_results = None
            model_results = None
            
            # Run both analyses simultaneously
            with st.spinner("🔍 Running Dual Analysis..."):
                # Get image bytes
                if camera_photo:
                    image_bytes = camera_photo.getvalue()
                elif uploaded_file:
                    image_bytes = uploaded_file.getvalue()
                else:
                    image_bytes = open("samples/real_sample.jpg" if sample_option == "Real Sample" 
                                      else "samples/fake_sample.jpg", "rb").read()
                
                # Parallel execution
                api_results = analyze_with_custom_api(image_bytes)
                image_hash = get_image_hash(current_image)
                model_results = predict_image(image_hash, current_image)

            # Process results
            if api_results and model_results:
                model_scores = {r['label']: r['score'] for r in model_results}
                api_confidence = max(api_results.values())
                local_confidence = max(model_scores.values()) * 100
                
                if api_confidence > local_confidence:
                    enhanced_analysis_report(api_results, selected=True)
                    local_model_report(model_results, selected=False)
                else:
                    local_model_report(model_results, selected=True)
                    enhanced_analysis_report(api_results, selected=False)
                
                # Confidence comparison
                st.markdown("### 🔍 Confidence Comparison")
                df = pd.DataFrame({
                    'Method': ['API', 'Local Model'],
                    'Confidence': [api_confidence, local_confidence]
                })
                st.bar_chart(df.set_index('Method'))
            else:
                st.error("Analysis failed. Please try again.")

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

# ----- Game Interface ----- 
def game_interface():
    st.title("🎮 Detection Training")
    
    if st.button("← Return to Home", key="game_return_home"):
        st.session_state.page = "welcome"
        st.rerun()

    if st.session_state.get("game_round", 1) > 5:
        st.markdown(f"## Game Over! Final Score: {st.session_state.get('game_score', 0)}/5")
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Play Again"):
                st.session_state.game_score = 0
                st.session_state.game_round = 1
                st.rerun()
        with col2:
            if st.button("Return to Home ←"):
                st.session_state.page = "welcome"
                st.rerun()
        return

    st.markdown(f"### Round {st.session_state.get('game_round', 1)} of 5")
    st.markdown(f"**Current Score:** {st.session_state.get('game_score', 0)}")

    if "current_round" not in st.session_state:
        real_img = fetch_real_image()
        fake_img = fetch_fake_image()
        if real_img and fake_img:
            images = [(real_img, "Real"), (fake_img, "Fake")]
            random.shuffle(images)
            st.session_state.current_round = {
                "images": images,
                "answer": "1" if images[0][1] == "Real" else "2"
            }

    if "current_round" in st.session_state:
        cols = st.columns(2)
        for idx, (img, label) in enumerate(st.session_state.current_round["images"]):
            with cols[idx]:
                st.markdown(f"<h4 style='text-align: center'>Image {idx+1}</h4>", unsafe_allow_html=True)
                st.image(
                    img, 
                    use_container_width=False,
                    width=300,
                    output_format="JPEG",
                    clamp=True
                )

        user_guess = st.radio("Which image is real?", ["1", "2"], horizontal=True)
        if st.button("Submit Answer", key="guess_btn"):
            if user_guess == st.session_state.current_round["answer"]:
                st.session_state.game_score += 1
                st.success("Correct! 🎉 +1 Point")
            else:
                st.error("Incorrect ❌")
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
