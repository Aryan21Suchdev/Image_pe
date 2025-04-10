import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

# 🎨 Page Config
st.set_page_config(page_title="Image Enhancer", layout="wide", page_icon="🎨")

# 🖌️ Custom CSS for colorful background
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background-image: linear-gradient(to right, #fdfbfb, #ebedee);
    }
    </style>
""", unsafe_allow_html=True)

# 💡 Image Processing Functions
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
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def apply_gaussian_noise(image, stddev=25):
    noise = np.random.normal(0, stddev, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def adjust_brightness(image, brightness=30):
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

def adjust_contrast(image, alpha=1.5):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def invert_colors(image):
    return cv2.bitwise_not(image)

def apply_sepia(image):
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

def apply_emboss(image):
    kernel = np.array([[ -2, -1, 0],
                       [ -1,  1, 1],
                       [  0,  1, 2]])
    return cv2.filter2D(image, -1, kernel)

def convert_image_to_download(image_array):
    img_pil = Image.fromarray(image_array)
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    return buffer.getvalue()

# 🎨 Title
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color:#FF4B4B;">🌟 Image Enhancement & Processing App 🌟</h1>
        <h4 style="color:#00BFFF;">Upload your image and transform it instantly!</h4>
    </div>
""", unsafe_allow_html=True)

# 🧭 Sidebar
st.sidebar.markdown("## 🎛️ Enhancement Controls")
uploaded_file = st.sidebar.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

technique = st.sidebar.selectbox("🎨 Choose an Enhancement", [
    "Smoothing",
    "Log Transformation",
    "Histogram Equalization",
    "Edge Detection",
    "Sharpening",
    "Gaussian Noise",
    "Brightness Adjustment",
    "Contrast Adjustment",
    "Color Inversion",
    "Sepia Filter",
    "Emboss Effect"
])

# ⚙️ Parameter Handling
if technique == "Smoothing":
    k = st.sidebar.slider("🔧 Kernel Size", 1, 21, 5, step=2)
elif technique == "Gaussian Noise":
    stddev = st.sidebar.slider("🌫️ Noise Intensity", 5, 100, 25, step=5)
elif technique == "Brightness Adjustment":
    brightness = st.sidebar.slider("💡 Brightness Level", -100, 100, 30, step=10)
elif technique == "Contrast Adjustment":
    alpha = st.sidebar.slider("⚡ Contrast Factor", 0.5, 3.0, 1.5, step=0.1)

# 🖼️ Display and Enhance
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.markdown("<h3 style='color:limegreen;'>📸 Original Image:</h3>", unsafe_allow_html=True)
    st.image(image_np, use_column_width=True)

    if st.sidebar.button("🚀 Apply Enhancement"):
        if technique == "Smoothing":
            output = apply_smoothing(image_np, k)
            caption = "🌀 Smoothed Image"
        elif technique == "Log Transformation":
            output = apply_log_transformation(image_np)
            caption = "📈 Log Transformed Image"
        elif technique == "Histogram Equalization":
            output = apply_histogram_equalization(image_np)
            caption = "📊 Histogram Equalized Image"
        elif technique == "Edge Detection":
            output = apply_edge_detection(image_np)
            caption = "🔍 Edge Detected Image"
        elif technique == "Sharpening":
            output = apply_sharpening(image_np)
            caption = "✨ Sharpened Image"
        elif technique == "Gaussian Noise":
            output = apply_gaussian_noise(image_np, stddev)
            caption = "🌫️ Gaussian Noise Added"
        elif technique == "Brightness Adjustment":
            output = adjust_brightness(image_np, brightness)
            caption = "💡 Brightness Adjusted"
        elif technique == "Contrast Adjustment":
            output = adjust_contrast(image_np, alpha)
            caption = "⚡ Contrast Adjusted"
        elif technique == "Color Inversion":
            output = invert_colors(image_np)
            caption = "🎭 Inverted Colors"
        elif technique == "Sepia Filter":
            output = apply_sepia(image_np)
            caption = "🎞️ Sepia Toned Image"
        elif technique == "Emboss Effect":
            output = apply_emboss(image_np)
            caption = "🪨 Embossed Image"

        # ✅ Show Processed Image
        st.markdown(f"<h3 style='color:#1E90FF;'>{caption}:</h3>", unsafe_allow_html=True)
        st.image(output, use_column_width=True, channels="BGR" if len(output.shape) == 3 else "GRAY")

        # 💾 Download Enhanced Image
        st.markdown("### 📥 Download Enhanced Image")
        img_bytes = convert_image_to_download(output)
        st.download_button(
            label="💾 Download as PNG",
            data=img_bytes,
            file_name="enhanced_image.png",
            mime="image/png"
        )
else:
    st.info("👈 Upload an image from the sidebar to get started!")
