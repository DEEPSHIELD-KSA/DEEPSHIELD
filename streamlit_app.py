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
from mtcnn import MTCNN  # Face detection library

# Initialize face detector
detector = MTCNN()

# ----- Face Detection Function -----
def detect_human_faces(image: Image.Image):
    """Detect human faces in image and return count and face locations"""
    try:
        # Convert PIL image to RGB array
        img_array = np.array(image.convert('RGB'))
        # Detect faces
        faces = detector.detect_faces(img_array)
        return len(faces), faces
    except Exception as e:
        st.error(f"Face detection error: {str(e)}")
        return 0, []

# ----- Modified Main Page -----
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
                    # Display original image
                    st.image(image, use_container_width=True, caption="Selected Image Preview")
                    
                    # Check for human faces before analysis
                    with st.spinner("🔍 Scanning for human faces..."):
                        face_count, faces = detect_human_faces(image)
                        
                        if face_count == 0:
                            st.error("❌ No human faces detected in this image. Please upload an image with at least one clear human face.")
                            return
                        else:
                            # Create a copy to draw face boxes
                            img_with_boxes = image.copy()
                            draw = ImageDraw.Draw(img_with_boxes)
                            
                            for face in faces:
                                x, y, width, height = face['box']
                                # Draw rectangle around face
                                draw.rectangle([(x, y), (x+width, y+height)], 
                                             outline="red", width=2)
                                # Draw face confidence
                                draw.text((x, y-10), 
                                         f"{face['confidence']:.2f}", 
                                         fill="red")
                            
                            st.image(img_with_boxes, 
                                    caption=f"Detected {face_count} face(s) (confidence shown)", 
                                    use_container_width=True)
                            
                            st.success(f"✅ Found {face_count} human face(s). Proceeding with analysis...")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    if (uploaded_file or sample_option != "Select") and 'image' in locals() and image is not None:
        # Only proceed if faces were detected
        if 'face_count' in locals() and face_count > 0:
            try:
                with st.spinner("🔬 Scanning image for AI fingerprints..."):
                    image_hash = get_image_hash(image)
                    result = predict_image(image_hash, image)
                    scores = {r["label"].lower(): r["score"] for r in result}
                
                st.markdown("---")
                st.markdown("### 📊 Detection Report")

                # Display face count metric
                col_face, col1, col2 = st.columns([1, 2, 2])
                with col_face:
                    st.metric("Faces Detected", face_count)
                
                with col1:
                    real_conf = scores.get("real", 0)
                    real_chart_data = pd.DataFrame({"Category": ["Real"], "Confidence": [real_conf]})
                    real_chart = (
                        alt.Chart(real_chart_data)
                        .mark_bar(size=40, color="#00ff88")
                        .encode(
                            x=alt.X("Category", title=""),
                            y=alt.Y("Confidence", title="Confidence", scale=alt.Scale(domain=[0, 1])),
                            tooltip=["Category", "Confidence"]
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(real_chart, use_container_width=True)

                with col2:
                    fake_conf = scores.get("fake", 0)
                    fake_chart_data = pd.DataFrame({"Category": ["Fake"], "Confidence": [fake_conf]})
                    fake_chart = (
                        alt.Chart(fake_chart_data)
                        .mark_bar(size=40, color="#ff4d4d")
                        .encode(
                            x=alt.X("Category", title=""),
                            y=alt.Y("Confidence", title="Confidence", scale=alt.Scale(domain=[0, 1])),
                            tooltip=["Category", "Confidence"]
                        )
                        .properties(height=200)
                    )
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
# REST OF YOUR CODE (keep all other existing functions)
# =======================

# Requirements note
st.sidebar.markdown("""
**Requirements:**  
`pip install mtcnn` for face detection
""")

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        welcome()
    elif st.session_state.page == "game":
        game()
    else:
        main()
