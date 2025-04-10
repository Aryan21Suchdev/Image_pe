import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Page Config
st.set_page_config(page_title="🎨 Image Enhancer", layout="wide", page_icon="🖼️")

# --- Image Processing Functions ---
def apply_smoothing(image, ksize):
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_log_transformation(image):
    image = np.array(image, dtype=np.float32) + 1
    log_transformed = np.log(image) * (255 / np.log(256))
    return np.clip(log_transformed, 0, 255).astype(np.uint8)

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
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def adjust_brightness(image, beta):
    return cv2.convertScaleAbs(image, beta=beta)

def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha)

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def invert_colors(image):
    return cv2.bitwise_not(image)

def add_gaussian_noise(image, mean=0, sigma=25):
    image = image.astype(np.float32)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# --- UI Header ---
st.markdown("""
    <h1 style='text-align: center; color: #FF6347;'>🖼️ Image Enhancement & Processing App</h1>
    <p style='text-align: center; font-size: 18px; color: #555;'>Upload an image and apply different enhancement techniques</p>
    <hr style="border: 1px solid #eee;">
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.markdown("## 🛠️ Controls")
uploaded_file = st.sidebar.file_uploader("📁 Upload an Image", type=["jpg", "png", "jpeg"])

technique = st.sidebar.selectbox("✨ Choose an Enhancement Technique", [
    "Smoothing", "Log Transformation", "Histogram Equalization", "Edge Detection", 
    "Sharpening", "Brightness Adjustment", "Contrast Adjustment", 
    "Rotation", "Color Inversion", "Add Gaussian Noise"
])

# Additional controls based on technique
if technique == "Smoothing":
    k = st.sidebar.slider("🔧 Kernel Size", 1, 21, 5, step=2)
elif technique == "Brightness Adjustment":
    beta = st.sidebar.slider("💡 Brightness", -100, 100, 0)
elif technique == "Contrast Adjustment":
    alpha = st.sidebar.slider("🎚️ Contrast", 0.5, 3.0, 1.0)
elif technique == "Rotation":
    angle = st.sidebar.slider("🔄 Rotation Angle", -180, 180, 0)
elif technique == "Add Gaussian Noise":
    sigma = st.sidebar.slider("📶 Noise Level (Sigma)", 1, 100, 25)

# --- Main Area ---
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
        elif technique == "Brightness Adjustment":
            output = adjust_brightness(image, beta)
            caption = "🌞 Brightness Adjusted Image"
        elif technique == "Contrast Adjustment":
            output = adjust_contrast(image, alpha)
            caption = "📏 Contrast Adjusted Image"
        elif technique == "Rotation":
            output = rotate_image(image, angle)
            caption = "🔄 Rotated Image"
        elif technique == "Color Inversion":
            output = invert_colors(image)
            caption = "🔁 Color Inverted Image"
        elif technique == "Add Gaussian Noise":
            output = add_gaussian_noise(image, sigma=sigma)
            caption = "🌫️ Gaussian Noise Added"

        st.markdown(f"<h3 style='color: #1E90FF;'>{caption}:</h3>", unsafe_allow_html=True)
        if technique in ["Histogram Equalization", "Edge Detection"]:
            st.image(output, use_column_width=True, channels="GRAY")
        else:
            st.image(output, use_column_width=True)
else:
    st.info("👈 Upload an image from the sidebar to get started!")

# Footer
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#999;'>Created with ❤️ using Streamlit</p>", unsafe_allow_html=True)


