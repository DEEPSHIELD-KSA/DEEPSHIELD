# Load models
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

# ----- Human Verification Model -----
@st.cache_resource
def load_human_model():
    processor = AutoProcessor.from_pretrained("prithivMLmods/Human-vs-NonHuman-Detection")
    model = AutoModelForImageClassification.from_pretrained("prithivMLmods/Human-vs-NonHuman-Detection")
    return processor, model

def is_human(image: Image.Image) -> bool:
    """Check if the image contains a human"""
    try:
        processor, model = load_human_model()
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        return model.config.id2label[predicted_class] == 'human'
    except Exception as e:
        st.error(f"Error in human verification: {str(e)}")
        return False

# ----- Helper functions for fetching images for the game -----
def fetch_real_image():
    # ... (keep existing fetch_real_image implementation unchanged) ...

def fetch_fake_image():
    # ... (keep existing fetch_fake_image implementation unchanged) ...

# ----- Rest of existing imports and helper functions -----
# ... (keep all existing imports and helper functions unchanged) ...

# ----- Modified Main Page with Human Verification -----
def main():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=800)
    except Exception as e:
        st.write("")

    st.title("Deepfake Detection System")

    with st.sidebar:
        # ... (keep existing sidebar code unchanged) ...

    col1, col2 = st.columns([4, 3])
    with col1:
        st.markdown("### 📤 Image Analysis Zone")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        sample_option = st.selectbox(
            "Or choose from samples:",
            ["Select", "Real Sample", "Fake Sample"],
            help="Explore pre-loaded examples to test the system"
        )
    with col2:
        if uploaded_file or sample_option != "Select":
            st.markdown("### 🔍 Preview")
            try:
                image = None
                if uploaded_file:
                    image = Image.open(uploaded_file)
                elif sample_option == "Real Sample":
                    if os.path.exists("samples/real_sample.jpg"):
                        image = Image.open("samples/real_sample.jpg")
                    else:
                        st.error("Real sample image not found. Please check 'samples/real_sample.jpg'")
                elif sample_option == "Fake Sample":
                    if os.path.exists("samples/fake_sample.jpg"):
                        image = Image.open("samples/fake_sample.jpg")
                    else:
                        st.error("Fake sample image not found. Please check 'samples/fake_sample.jpg'")

                if image:
                    st.image(image, use_container_width=True, caption="Selected Image Preview")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    if (uploaded_file or sample_option != "Select") and 'image' in locals() and image is not None:
        try:
            with st.spinner("🔬 Scanning image for AI fingerprints..."):
                # Human verification check
                if not is_human(image):
                    st.error("🚫 Non-human image detected! Please upload an image containing a human face.")
                    return

                # Proceed with deepfake detection if human
                image_hash = get_image_hash(image)
                result = predict_image(image_hash, image)
                scores = {r["label"].lower(): r["score"] for r in result}
            
            st.markdown("---")
            st.markdown("### 📊 Detection Report")

            # ... (keep existing visualization code unchanged) ...

        except Exception as e:
            st.error(f"🔧 Analysis error: {str(e)}")

# ----- Rest of the code remains unchanged -----
# ... (keep all other functions and page routing code unchanged) ...

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        welcome()
    elif st.session_state.page == "game":
        game()
    else:
        main()
