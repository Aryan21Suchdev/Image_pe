import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# 🎨 Page Configuration
st.set_page_config(page_title="Image Enhancer", layout="wide", page_icon="🎨")

# 🌈 Background CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #fdfbfb, #ebedee);
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Processing Functions
def to_rgb(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_bgr(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def apply_smoothing(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_log_transformation(image):
    image = np.array(image, dtype=np.float32) + 1
    log_transformed = np.log(image) * (255 / np.log(256))
    return np.clip(log_transformed, 0, 255).astype(np.uint8)

def apply_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_gaussian_noise(image, stddev=25):
    noise = np.random.normal(0, stddev, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255)
    return noisy.astype(np.uint8)

def adjust_brightness(image, brightness=30):
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

def adjust_contrast(image, alpha=1.5):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def invert_colors(image):
    return cv2.bitwise_not(image)

def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, sepia_filter)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_emboss(image):
    kernel = np.array([[ -2, -1, 0],
                       [ -1,  1, 1],
                       [  0,  1, 2]])
    return cv2.filter2D(image, -1, kernel)

def convert_image_to_download(image_array):
    img_rgb = Image.fromarray(image_array)
    buffer = BytesIO()
    img_rgb.save(buffer, format="PNG")
    return buffer.getvalue()

# 🌟 Title
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color:#FF4B4B;">🎨 Image Enhancement App</h1>
        <h4 style="color:#0080ff;">Upload and apply stunning visual transformations</h4>
    </div>
""", unsafe_allow_html=True)

# 🚪 Sidebar Input
st.sidebar.header("🧰 Controls")
uploaded_file = st.sidebar.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

technique = st.sidebar.selectbox("🛠️ Select Enhancement", [
    "Smoothing", "Log Transformation", "Histogram Equalization", "Edge Detection",
    "Sharpening", "Gaussian Noise", "Brightness Adjustment", "Contrast Adjustment",
    "Color Inversion", "Sepia Filter", "Emboss Effect"
])

# 🎚️ Technique Parameters
params = {}
if technique == "Smoothing":
    params["ksize"] = st.sidebar.slider("🌀 Kernel Size", 1, 21, 5, step=2)
elif technique == "Gaussian Noise":
    params["stddev"] = st.sidebar.slider("🌫️ Noise Std Dev", 5, 100, 25)
elif technique == "Brightness Adjustment":
    params["brightness"] = st.sidebar.slider("💡 Brightness Level", -100, 100, 30)
elif technique == "Contrast Adjustment":
    params["alpha"] = st.sidebar.slider("⚡ Contrast Level", 0.5, 3.0, 1.5)

# 🎯 Main Display
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_bgr = to_bgr(image)

    st.markdown("### 📸 Original Image")
    st.image(image, use_column_width=True)

    if st.sidebar.button("✨ Enhance Image"):
        if technique == "Smoothing":
            output = apply_smoothing(image_bgr, params["ksize"])
        elif technique == "Log Transformation":
            output = apply_log_transformation(image_bgr)
        elif technique == "Histogram Equalization":
            output = apply_histogram_equalization(image_bgr)
        elif technique == "Edge Detection":
            output = apply_edge_detection(image_bgr)
        elif technique == "Sharpening":
            output = apply_sharpening(image_bgr)
        elif technique == "Gaussian Noise":
            output = apply_gaussian_noise(image_bgr, params["stddev"])
        elif technique == "Brightness Adjustment":
            output = adjust_brightness(image_bgr, params["brightness"])
        elif technique == "Contrast Adjustment":
            output = adjust_contrast(image_bgr, params["alpha"])
        elif technique == "Color Inversion":
            output = invert_colors(image_bgr)
        elif technique == "Sepia Filter":
            output = apply_sepia(image_bgr)
        elif technique == "Emboss Effect":
            output = apply_emboss(image_bgr)

        # Convert for correct color display
        if len(output.shape) == 2:
            output_rgb = to_rgb(output)
        else:
            output_rgb = to_rgb(output)

        st.markdown("### 🌈 Enhanced Image")
        st.image(output_rgb, use_column_width=True)

        # 💾 Download
        st.markdown("### 📥 Download Your Image")
        img_bytes = convert_image_to_download(output_rgb)
        st.download_button("💾 Download PNG", data=img_bytes, file_name="enhanced_image.png", mime="image/png")
else:
    st.info("📎 Upload an image from the sidebar to get started!")
