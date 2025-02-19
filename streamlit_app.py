import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd  # Import pandas for DataFrame
import altair as alt  # Import Altair for custom bar chart colors
import requests  # For fetching fake images
import os  # For accessing local files
import random  # For randomizing image positions


st.set_page_config(page_title="Deepfake Detection", page_icon="🖼️", layout="centered")

# Load the deepfake detection model
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    return pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

# fetch a fake image from thispersondoesnotexist.com
def fetch_fake_image():
    try:
        response = requests.get("https://thispersondoesnotexist.com/", stream=True)
        if response.status_code == 200:
            img = Image.open(response.raw)
            return img.resize((512, 512))  # Resize to 512x512
    except Exception as e:
        st.error(f"Error fetching fake image: {e}")
    return None

# Function to load a random real image from the game_real folder
def fetch_real_image():
    if "used_real_images" not in st.session_state:
        st.session_state.used_real_images = set()  # Track used real images

    real_images = os.listdir("game_real")  # List all files in the game_real folder
    available_images = [img for img in real_images if img not in st.session_state.used_real_images]

    # If we've used all images, reset the used list so images can repeat
    if not available_images:
        st.session_state.used_real_images = set()
        available_images = real_images

    random_image = random.choice(available_images)  # Choose a random unused image
    st.session_state.used_real_images.add(random_image)  # Mark the image as used
    image = Image.open(os.path.join("game_real", random_image))
    return image.resize((512, 512))  # Resize to 512x512

# Streamlit App - Main Page
def main():
    # Add a logo at the top center
    st.image("logo.png", use_container_width=True)  # Replace with your logo path

    # Add a title and description
    st.title("Deepfake Detection System")

    # Sidebar for navigation
    st.sidebar.title("Are You Smarter Than AI?!")
    if st.sidebar.button("Go to Game"):
        st.session_state.page = "game"
        st.rerun()

    # Short description under the image
    st.write(
        """
        This project uses advanced AI models to detect deepfake images.\n
        Upload an image or choose a sample to see if it's real or AI-generated.
        """
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Sample image dropdown
    sample_option = st.selectbox("Or choose a sample image:", ["Select", "Real Sample", "Fake Sample"])
    
    if sample_option == "Real Sample":
        image = Image.open("samples/real_sample.jpg")  # Replace with your real sample image
        st.image(image, caption="Real Sample Image", use_container_width=True)
    elif sample_option == "Fake Sample":
        image = Image.open("samples/fake_sample.jpg")  # Replace with your fake sample image
        st.image(image, caption="Fake Sample Image", use_container_width=True)
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if uploaded_file is not None or sample_option != "Select":
        try:
            # Show a progress bar while analyzing
            with st.spinner("Analyzing the image..."):
                # Load model and predict
                model = load_model()
                result = model(image)
                
                # Extract confidence scores for "Real" and "Fake"
                scores = {r["label"].lower(): r["score"] for r in result}  # Convert labels to lowercase
            
            # Display results in a modern layout
            st.success("Analysis Complete!")
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Real Confidence", value=f"{scores.get('real', 0) * 100:.2f}%")
            with col2:
                st.metric(label="Fake Confidence", value=f"{scores.get('fake', 0) * 100:.2f}%")
            
            # Display confidence scores as a vertical bar chart with custom colors
            st.write("### Confidence Scores")
            chart_data = pd.DataFrame({
                "Category": ["Real", "Fake"],
                "Confidence": [scores.get("real", 0), scores.get("fake", 0)],
            })
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X("Category", sort=["Real", "Fake"]),
                y="Confidence",
                color=alt.Color("Category", scale=alt.Scale(domain=["Real", "Fake"], range=["#00FF00", "#FF0000"])),
            )
            st.altair_chart(chart, use_container_width=True)
            
            # Determine the final prediction
            final_prediction = max(scores, key=scores.get)
            final_confidence = scores[final_prediction]
            st.write("### Final Prediction")
            if final_prediction == "fake":
                st.error(f"⚠️ This image is likely AI-generated (deepfake) with {final_confidence * 100:.2f}% confidence.")
            else:
                st.success(f"✅ This image is likely real with {final_confidence * 100:.2f}% confidence.")
        
        except Exception as e:
            st.error(f"An error occurred: {e}. Please upload a valid image.")

# Streamlit App - Game Page
def game():
    st.title("Deepfake Game")
    st.write("Guess which image is real! You have 5 rounds.")

    # Initialize game state if not already present
    if "game_score" not in st.session_state:
        st.session_state.game_score = 0
    if "game_round" not in st.session_state:
        st.session_state.game_round = 1

    # When the game is over, show final score
    if st.session_state.game_round > 5:
        st.write(f"**Game Over! Your score: {st.session_state.game_score}/5**")
        if st.button("Play Again"):
            st.session_state.game_score = 0
            st.session_state.game_round = 1
            st.session_state.used_real_images = set()  # Reset used images
            st.session_state.pop("current_round_data", None)
            st.session_state.pop("round_submitted", None)
            st.session_state.pop("round_result", None)
            st.rerun()
        return

    st.write(f"Round {st.session_state.game_round} of 5")

    # Create or load the current round's data
    if "current_round_data" not in st.session_state:
        # Fetch one real and one fake image
        real_image = fetch_real_image()
        fake_image = fetch_fake_image()
        if real_image is None or fake_image is None:
            st.error("Failed to load images. Please try again.")
            return

        # Randomize the position of the images
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

    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.current_round_data["left_image"], caption="Left Image", use_container_width=True)
    with col2:
        st.image(st.session_state.current_round_data["right_image"], caption="Right Image", use_container_width=True)

    # Use a placeholder container for the input controls (radio and submit button)
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
                control_container.empty()  # Remove the radio and submit button immediately

    # If the answer has been submitted, display the result and a Next Round button
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
            st.rerun()

    # Always show a button to go back to the home page
    if st.button("Go to Home"):
        st.session_state.page = "main"
        st.rerun()

# Page routing
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main()
elif st.session_state.page == "game":
    game()
