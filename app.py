import streamlit as st
import requests
import io
from PIL import Image

# FastAPI endpoint URL
API_URL = "http://localhost:8000/predict/"  # Make sure your FastAPI server is running

# Set page title and layout
st.set_page_config(page_title="Happy vs Sad Image Classifier", page_icon=":smiley:", layout="centered")

# Add custom header with an emoji
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Happy vs Sad Image Classifier</h1>", unsafe_allow_html=True)

# Add a playful description
st.markdown("""
    <p style='text-align: center; color: #555555; font-size: 16px;'>Upload a photo to find out if the person is happy or sad! 😊</p>
    <p style='text-align: center; color: #555555; font-size: 14px;'>Powered by Deep Learning and FastAPI</p>
""", unsafe_allow_html=True)

# Add an emoji for upload section
st.subheader(":camera: Upload an image to classify:")

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to bytes for API call
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Send request to FastAPI backend
    with st.spinner("Predicting... please wait..."):
        response = requests.post(API_URL, files={"file": img_bytes})

    if response.status_code == 200:
        # Display result with a custom emoji
        result = response.json()
        label = result["label"]
        prediction_score = result["prediction"]
        
        # Stylish results
        if label == "Sad":
            st.markdown(f"<h3 style='text-align: center; color: #FF6347;'>Prediction: {label} 😞</h3>", unsafe_allow_html=True)
        elif label == "Happy":
            st.markdown(f"<h3 style='text-align: center; color: #FFD700;'>Prediction: {label} 😁</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center; color: #FF8C00;'>Prediction: {label} 🤔</h3>", unsafe_allow_html=True)

        st.markdown(f"Confidence Score: {prediction_score:.2f}", unsafe_allow_html=True)
    else:
        st.error("Error in prediction. Please try again.")

# Footer with social media links or credits
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #888888;'>Built with ❤️ using Streamlit, FastAPI, and Deep Learning</p>
    <p style='text-align: center; color: #888888;'>Connect with us on [GitHub](https://github.com/your-profile) and [LinkedIn](https://www.linkedin.com/in/your-profile/)</p>
""", unsafe_allow_html=True)
