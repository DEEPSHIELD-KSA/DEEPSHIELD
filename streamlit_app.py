# ... (keep all previous imports and helper functions)

# =======================
# Welcome Page
# =======================
def welcome():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=300)
    except Exception as e:
        st.write("")

    st.title("Deepfake Detective 🕵️")

    st.markdown("""
    ## Welcome to the Deepfake Detection Project

    This application helps you detect AI-generated images using state-of-the-art machine learning models. 
    Explore the capabilities of deepfake detection through image analysis or test your skills in the detection challenge game!
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Key Features
        - **Image Analysis**: Upload an image to check if it's real or AI-generated
        - **Detection Game**: Train your eye to spot deepfakes in a fun, interactive game
        - **Real-time Predictions**: Get instant results with confidence scores
        - **Educational Insights**: Learn about deepfake technology and detection methods
        """)
    
    with col2:
        st.markdown("""
        ### 🕹️ How to Use
        1. Use the **Image Analysis** page to check individual images
        2. Try the **Detection Game** to test your detection skills
        3. Review the confidence metrics to understand model predictions
        4. Explore different sample images to see varied results
        """)

    st.markdown("---")
    st.markdown("### 📝 Project Notes")
    st.markdown("""
    **Technical Details**:
    - Built with 🤗 Transformers and Streamlit
    - Uses a fine-tuned ViT (Vision Transformer) model
    - Model trained on diverse dataset of real and AI-generated images
    - Continuous updates to improve accuracy and reliability

    **Ethical Considerations**:
    - Intended for educational purposes only
    - Results should not be considered definitive proof of authenticity
    - AI detection systems can have inherent biases
    - Always verify critical content through multiple methods

    **Future Directions**:
    - Video analysis capabilities
    - Detailed explainability features
    - Multi-model consensus system
    - Real-time webcam analysis
    """)

    if st.button("Get Started →", use_container_width=True, type="primary"):
        st.session_state.page = "main"
        rerun()

# =======================
# Main Page: Image Analysis (updated with return to welcome)
# =======================
def main():
    try:
        logo_image = Image.open("logo.png")
        st.image(logo_image, width=300)
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
            if "current_round_data" in st.session_state:
                st.session_state.pop("current_round_data")
            if "round_submitted" in st.session_state:
                st.session_state.pop("round_submitted")
            if "round_result" in st.session_state:
                st.session_state.pop("round_result")
            st.session_state.page = "game"
            rerun()

    # ... (rest of main page content remains the same)

# =======================
# Page Routing (updated)
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
