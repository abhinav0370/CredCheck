# nmain.py

import streamlit as st
import os
from langdetect import detect
from audio_to_text import audio_to_text
from convert_to_english import translation
from cred_check import fake_news_detector
from claimbuster_check import check_claim
from top_headlines import fetch_headlines
from img_to_text import extract_text_from_image
from video_to_audio import extract_audio
from PIL import Image

# ---------------------------
# 8. Helper Functions
# ---------------------------
def is_english(text):
    """Check if the given text is in English."""
    try:
        return detect(text) == 'en'
    except:
        return False

# Define threshold for fake news classification
SCORE_THRESHOLD = 0.5

def classify_claim(score):
    """Classify the claim based on the score."""
    return "🔴 Fake" if score < SCORE_THRESHOLD else "🟢 Real"

def classify_auth(is_fake):
    """Classify the authentication result."""
    return "🔴 Fake" if is_fake else "🟢 Real"

# ---------------------------
# 1. Page Configuration
# ---------------------------
st.set_page_config(
    page_title="🔍 CredCheck",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# 2. Custom CSS for Layout and Styling
# ---------------------------
custom_css = """
<style>
/* Increased Margins for Improved Readability */
.block-container {
    padding-left: 10rem !important;
    padding-right: 10rem !important;
}

/* Responsive Design: Adjust padding based on screen width */
@media (max-width: 1200px) {
    .block-container {
        padding-left: 5rem !important;
        padding-right: 5rem !important;
    }
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
}

/* Hide Streamlit's Default Menu and Footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Header Styling */
.header {
    background-color: #4CAF50;
    padding: 20px;
    text-align: center;
    position: fixed;
    width: 84%;
    top: 60px;
    left: 50%;
    transform: translateX(-50%);
    border-radius: 10px;
    z-index: 1000;
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: 5%;
}

.header .title {
    font-size: 32px;
    color: white;
    font-weight: bold;
}

.header .nav-links a {
    color: white;
    margin-left: 30px;
    text-decoration: none;
    font-size: 20px;
}

.header .nav-links a:hover {
    text-decoration: underline;
}

/* Hero Section Styling */
.hero {
    margin-top: 100px; /* Adjusted to accommodate fixed header */
    text-align: center;
}

.hero img {
    width: 90%;
    max-width: 900px;
    border-radius: 10px;
}

/* Content Section Styling */
.content {
    margin-top: 50px;
    text-align: center;
}

.content .input-methods {
    margin-top: 30px;
}

.content .input-methods button {
    margin: 3px;
    padding: 2px 3px;
    font-size: 18px;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    background-color: #4CAF50;
    color: white;
    transition: background-color 0.3s ease, transform 0.2s ease;
}

.content .input-methods button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}

/* Plotly Section Adjustments */
.plotly-section {
    margin-top: 50px;
}

.plotly-section .metrics {
    margin-top: 40px;
}
</style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------
# 3. Header Section with Navigation Link to Metrics
# ---------------------------
header = """
<div class="header">
    <div class="title">CredCheck</div>
    <div class="nav-links">
        <a href="#news-check">News Check</a>
        <a href="#real-time-analysis">Real Time Analysis</a>
    </div>
</div>
"""
st.markdown(header, unsafe_allow_html=True)



# ---------------------------
# 4. Hero Section using st.image with use_container_width
# ---------------------------
st.markdown('<div class="hero"></div>', unsafe_allow_html=True)

# Load and display the image
image_path = os.path.join("images", "img1.png")  # Ensure the path is correct
try:
    hero_image = Image.open(image_path)
    st.image(hero_image, use_container_width=True)  # Updated parameter
except FileNotFoundError:
    st.error(f"Image not found at path: {image_path}. Please ensure the image exists.")

# ---------------------------
# 5. Content Section - News Check
# ---------------------------
st.markdown('<div id="news-check"></div>', unsafe_allow_html=True)

with st.container():
    st.markdown("<h2 style='color:#333333;'>📰 News Check</h2>", unsafe_allow_html=True)
    
    # Text Input for News Headline
    headline = st.text_input("📝 Enter the news headline:", key="headline_input")
    
    # Four Rounded Buttons for Different Input Methods
    st.markdown('<div class="input-methods">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Text", key="text_input"):
            st.session_state.input_type = "Text"
    
    with col2:
        if st.button("Image", key="image_input"):
            st.session_state.input_type = "Image"
    
    with col3:
        if st.button("Audio", key="audio_input"):
            st.session_state.input_type = "Audio"
    
    with col4:
        if st.button("Video", key="video_input"):
            st.session_state.input_type = "Video"
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle Input Methods
    input_type = st.session_state.get('input_type', 'Text')
    
    # Analyze Button with Conditional Rendering
    if input_type == "Text":
        if st.button("Analyze", key="analyze_text"):
            if headline:
                with st.spinner('📊 Processing...'):
                    st.info("📊 Analyzing headline with CredCheck...")
                    auth_result = fake_news_detector(headline)
                    st.markdown(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                    
                    # Only use ClaimBuster if flagged as fake
                    if auth_result.get('is_fake', False):
                        if not is_english(headline):
                            st.info("🔄 Translating to English for External API")
                            translated_headline = translation(headline)
                            st.success(f"**Translated Text:** {translated_headline}")
                        else:
                            translated_headline = headline
                        st.info("🔍 Verifying Flagged Content With External API")
                        claimbuster_result = check_claim(translated_headline)
                        
                        if "error" in claimbuster_result:
                            st.error(claimbuster_result["error"])
                        else:
                            for result in claimbuster_result["results"]:
                                score = result["score"]
                                classification = classify_claim(score)
                                st.markdown(f"**Claim:** {result['text']}")
                                st.markdown(f"**Score:** {score}")
                                st.markdown(f"**ClaimBuster Classification:** {classification}")
                                st.markdown("---")
                    else:
                        st.success("✅ The news is classified as real.")
            else:
                st.error("⚠️ Please enter a headline.")

    elif input_type == "Audio":
        audio_file = st.file_uploader("🎤 Upload an audio file", type=["wav", "mp3", "flac"], key="audio_uploader")
        if st.button("Analyze", key="analyze_audio"):
            if audio_file:
                with st.spinner('🎧 Processing audio...'):
                    temp_audio_path = "temp_audio.wav"
                    with open(temp_audio_path, "wb") as f:
                        f.write(audio_file.getbuffer())
                    text = audio_to_text(temp_audio_path)
                    os.remove(temp_audio_path)
                    st.markdown(f"**Transcribed Text:** {text}")
                    st.info("📊 Analyzing text with CredCheck...")
                    auth_result = fake_news_detector(text)
                    st.markdown(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                    
                    # Only use ClaimBuster if flagged as fake
                    if auth_result.get('is_fake', False):
                        if not is_english(text):
                            st.info("🔄 Translating to English for External API")
                            translated_text = translation(text)
                            st.success(f"**Translated Text:** {translated_text}")
                        else:
                            translated_text = text
                        st.info("🔍 Verifying Flagged Content With External API")
                        claimbuster_result = check_claim(translated_text)
                        
                        if "error" in claimbuster_result:
                            st.error(claimbuster_result["error"])
                        else:
                            for result in claimbuster_result["results"]:
                                score = result["score"]
                                classification = classify_claim(score)
                                st.markdown(f"**Claim:** {result['text']}")
                                st.markdown(f"**Score:** {score}")
                                st.markdown(f"**ClaimBuster Classification:** {classification}")
                                st.markdown("---")
                    else:
                        st.success("✅ The news is classified as real.")
            else:
                st.error("⚠️ Please upload an audio file.")

    elif input_type == "Image":
        image_file = st.file_uploader("🖼️ Upload an image file", type=["png", "jpg", "jpeg", "tiff"], key="image_uploader")
        if st.button("Analyze", key="analyze_image"):
            if image_file:
                with st.spinner('🖼️ Processing image...'):
                    temp_image_path = "temp_image"
                    with open(temp_image_path, "wb") as f:
                        f.write(image_file.getbuffer())
                    extracted_text = extract_text_from_image(temp_image_path)
                    os.remove(temp_image_path)
                    st.markdown(f"**Extracted Text:** {extracted_text}")
                    
                    if extracted_text.startswith("Error:"):
                        st.error(extracted_text)
                    else:
                        st.info("📊 Analyzing text with CredCheck...")
                        auth_result = fake_news_detector(extracted_text)
                        st.markdown(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                        
                        # Only use ClaimBuster if flagged as fake
                        if auth_result.get('is_fake', False):
                            if not is_english(extracted_text):
                                st.info("🔄 Translating to English for External API")
                                translated_text = translation(extracted_text)
                                st.success(f"**Translated Text:** {translated_text}")
                            else:
                                translated_text = extracted_text
                            st.info("🔍 Verifying Flagged Content With External API")
                            claimbuster_result = check_claim(translated_text)
                            
                            if "error" in claimbuster_result:
                                st.error(claimbuster_result["error"])
                            else:
                                for result in claimbuster_result["results"]:
                                    score = result["score"]
                                    classification = classify_claim(score)
                                    st.markdown(f"**Claim:** {result['text']}")
                                    st.markdown(f"**Score:** {score}")
                                    st.markdown(f"**ClaimBuster Classification:** {classification}")
                                    st.markdown("---")
                        else:
                            st.success("✅ The news is classified as real.")
            else:
                st.error("⚠️ Please upload an image file.")

    elif input_type == "Video":
        video_file = st.file_uploader("🎥 Upload a video file", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
        if st.button("Analyze", key="analyze_video"):
            if video_file:
                with st.spinner('🎥 Processing video...'):
                    temp_video_path = "temp_video.mp4"
                    with open(temp_video_path, "wb") as f:
                        f.write(video_file.getbuffer())
                    extracted_audio_path = extract_audio(temp_video_path)
                    if extracted_audio_path.startswith("Error:"):
                        st.error(extracted_audio_path)
                        os.remove(temp_video_path)
                    else:
                        text = audio_to_text(extracted_audio_path)
                        st.markdown(f"**Transcribed Text from Video:** {text}")
                        os.remove(extracted_audio_path)
                        os.remove(temp_video_path)
                        st.info("📊 Analyzing text with CredCheck...")
                        auth_result = fake_news_detector(text)
                        st.markdown(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                        
                        # Only use ClaimBuster if flagged as fake
                        if auth_result.get('is_fake', False):
                            if not is_english(text):
                                st.info("🔄 Translating to English for External API")
                                translated_text = translation(text)
                                st.success(f"**Translated Text:** {translated_text}")
                            else:
                                translated_text = text
                            st.info("🔍 Verifying Flagged Content With External API")
                            claimbuster_result = check_claim(translated_text)
                            
                            if "error" in claimbuster_result:
                                st.error(claimbuster_result["error"])
                            else:
                                for result in claimbuster_result["results"]:
                                    score = result["score"]
                                    classification = classify_claim(score)
                                    st.markdown(f"**Claim:** {result['text']}")
                                    st.markdown(f"**Score:** {score}")
                                    st.markdown(f"**ClaimBuster Classification:** {classification}")
                                    st.markdown("---")
                        else:
                            st.success("✅ The news is classified as real.")
            else:
                st.error("⚠️ Please upload a video file.")

# ---------------------------
# 6. Secondary Content - Real Time News Analysis
# ---------------------------
# Add an HTML anchor for Real Time Analysis
st.markdown('<div id="real-time-analysis"></div>', unsafe_allow_html=True)

# Separate Top Headlines Section
st.header("📰 Real Time News Analysis")
if st.button("Fetch Top Headlines"):
    with st.spinner('📥 Fetching top headlines...'):
        headlines = fetch_headlines()
        if isinstance(headlines, list):
            st.write("Fetched headlines:")
            for idx, headline in enumerate(headlines, start=1):
                st.write(f"{idx}. {headline}")
            
            # Set number of headlines to analyze to ten
            num_to_analyze = min(10, len(headlines))
            headlines_to_analyze = headlines[:num_to_analyze]

            # Analyze each headline with CredCheck first
            with st.spinner('📊 Analyzing headlines with CredCheck...'):
                for idx, headline in enumerate(headlines_to_analyze, start=1):
                    st.write(f"**Headline {idx}:** {headline}")
                    
                    # CredCheck Fake News Detection
                    auth_result = fake_news_detector(headline)
                    st.write(f"**CredCheck Classification:** {classify_auth(auth_result.get('is_fake', False))}")
                    
                    # Only use ClaimBuster if flagged as fake
                    if auth_result.get('is_fake', False):
                        st.info("🔍 Verifying Flagged Content With External API")
                        claimbuster_result = check_claim(headline)
                        if "error" in claimbuster_result:
                            st.error(f"Error analyzing headline {idx} with ClaimBuster: {claimbuster_result['error']}")
                        else:
                            for result in claimbuster_result["results"]:
                                score = result["score"]
                                classification = classify_claim(score)
                                st.write(f"**Claim:** {result['text']}")
                                st.write(f"**Score:** {score}")
                                st.write(f"**ClaimBuster Classification:** {classification}")
                    else:
                        st.success("✅ The news is classified as real.")
                    st.markdown("---")
        else:
            st.error(headlines.get("error", "⚠️ An error occurred."))
# ---------------------------
# 7. Footer
# ---------------------------
st.markdown("---")
st.markdown("Developed by [Team Ignite](https://github.com/yourusername)")
