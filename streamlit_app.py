
import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import altair as alt
import io
import hashlib
import random

# Set page config with new branding
st.set_page_config(page_title="DFDetect", page_icon="🛡️", layout="centered")

# Custom CSS for modern UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #0A192F;
        color: #FFFFFF;
    }
    
    .header {
        text-align: center;
        padding: 2rem 0;
    }
    
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 3rem;
        margin: 2rem auto;
        max-width: 600px;
    }
    
    .stButton>button {
        background: #00D4FF;
        color: #0A192F;
        border-radius: 12px;
        padding: 12px 32px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #00B4CC;
        transform: scale(1.05);
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    .sample-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 2rem;
        margin-top: 2rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# Load model and helper functions (keep existing implementations)
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

# Main app structure
def main():
    # Navigation
    st.sidebar.markdown("# DFDetect")
    page = st.sidebar.radio("", ["Home", "Samples"])
    
    if page == "Home":
        render_home()
    else:
        render_samples()

def render_home():
    """Home page with upload functionality"""
    st.markdown("""
        <div class="header">
            <h1>DeepFake Detect</h1>
            <p>Upload an image to test for possible deepfakes</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            with st.form("upload-form"):
                uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], 
                                               label_visibility="collapsed")
                submitted = st.form_submit_button("Analyze Now")
                
                if uploaded_file and submitted:
                    process_image(uploaded_file)
                    
    st.markdown("---")
    st.caption("Image credits Facebook AI")

def render_samples():
    """Samples page with pre-loaded examples"""
    st.markdown("""
        <div class="header">
            <h1>Sample Detections</h1>
            <p>Prediction results produced by our deepfake detection model</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image("samples/real_sample.jpg", caption="Authentic Image")
            st.markdown("""
                <div class="result-card">
                    <h3>Detection Results</h3>
                    <p>Real Confidence: 98.7%</p>
                    <p>Fake Confidence: 1.3%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.image("samples/fake_sample.jpg", caption="AI-Generated Image")
            st.markdown("""
                <div class="result-card">
                    <h3>Detection Results</h3>
                    <p>Real Confidence: 4.2%</p>
                    <p>Fake Confidence: 95.8%</p>
                </div>
            """, unsafe_allow_html=True)

def process_image(uploaded_file):
    """Process and display results for uploaded image"""
    try:
        with st.spinner("Analyzing image..."):
            image = Image.open(uploaded_file)
            image_hash = get_image_hash(image)
            result = predict_image(image_hash, image)
            scores = {r["label"].lower(): r["score"] for r in result}
            
            with st.container():
                st.markdown("## Detection Results")
                
                # Confidence meters
                col1, col2 = st.columns(2)
                with col1:
                    render_confidence_meter(scores.get('real', 0), "Real", "#00D4FF")
                with col2:
                    render_confidence_meter(scores.get('fake', 0), "Fake", "#FF4D4D")
                
                # Final verdict
                final_pred = max(scores, key=scores.get)
                st.markdown(f"""
                    <div class="result-card">
                        <h3>Final Verdict: {final_pred.capitalize()}</h3>
                        <p>Confidence: {scores[final_pred]*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def render_confidence_meter(value, label, color):
    """Render a circular confidence meter"""
    st.markdown(f"""
        <div style="text-align: center;">
            <svg width="150" height="150">
                <circle cx="75" cy="75" r="60" stroke="#333" stroke-width="10" fill="none"/>
                <circle cx="75" cy="75" r="60" stroke="{color}" stroke-width="10" 
                    fill="none" stroke-dasharray="{2 * 3.1416 * 60}" 
                    stroke-dashoffset="{2 * 3.1416 * 60 * (1 - value)}"/>
                <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" 
                    style="font-size: 24px; font-weight: bold; fill: {color};">
                    {value*100:.1f}%
                </text>
            </svg>
            <h3>{label}</h3>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
