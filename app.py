import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set Page Config
st.set_page_config(page_title="Image Enhancer", layout="wide")

# Image Processing Functions
def apply_smoothing(image, ksize):
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_log_transformation(image):
    image = np.array(image, dtype=np.float32) + 1
    log_transformed = np.log(image) * (255 / np.log(256))
    log_transformed = np.clip(log_transformed, 0, 255)
    return np.array(log_transformed, dtype=np.uint8)

def apply_histogram_equalization(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image)

def apply_edge_detection(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(image, 100, 200)

def apply_sharpening(image):
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# UI Title
st.markdown(
    "<h1 style='text-align: center; color: #FF6347;'>🎨 Image Enhancement & Processing App</h1>", 
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🛠️ Controls")

uploaded_file = st.sidebar.file_uploader("📁 Upload an Image", type=["jpg", "png", "jpeg"])

technique = st.sidebar.selectbox(
    "✨ Choose an Enhancement Technique",
    ["Smoothing", "Log Transformation", "Histogram Equalization", "Edge Detection", "Sharpening"]
)

# Optional parameter for smoothing
if technique == "Smoothing":
    k = st.sidebar.slider("🔧 Kernel Size", min_value=1, max_value=21, step=2, value=5)

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.markdown("<h3 style='color: green;'>📸 Original Image:</h3>", unsafe_allow_html=True)
    st.image(image, use_column_width=True)

    if st.sidebar.button("🚀 Apply Enhancement"):
        if technique == "Smoothing":
            output = apply_smoothing(image, k)
            caption = "🌀 Smoothed Image"
        elif technique == "Log Transformation":
            output = apply_log_transformation(image)
            caption = "📈 Log Transformed Image"
        elif technique == "Histogram Equalization":
            output = apply_histogram_equalization(image)
            caption = "📊 Histogram Equalized Image"
        elif technique == "Edge Detection":
            output = apply_edge_detection(image)
            caption = "🔍 Edge Detected Image"
        elif technique == "Sharpening":
            output = apply_sharpening(image)
            caption = "✨ Sharpened Image"

        # Display processed image
        st.markdown(f"<h3 style='color: #1E90FF;'>{caption}:</h3>", unsafe_allow_html=True)
        if technique in ["Histogram Equalization", "Edge Detection"]:
            st.image(output, use_column_width=True, channels="GRAY")
        else:
            st.image(output, use_column_width=True)
else:
    st.info("👈 Upload an image from the sidebar to get started!")

