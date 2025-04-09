import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Image Processing Functions
def apply_smoothing(image, ksize):
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_log_transformation(image):
    image = np.array(image, dtype=np.float32) + 1  # Avoid log(0)
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
    # Sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

# Streamlit UI
st.title("Image Enhancement & Processing")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.image(image, caption="Original Image", use_column_width=True)

    option = st.selectbox("Choose an enhancement technique",
                          ["Smoothing", "Log Transformation", "Histogram Equalization", "Edge Detection", "Sharpening"])

    # Optional smoothing slider
    if option == "Smoothing":
        k = st.slider("Smoothing Kernel Size", min_value=1, max_value=21, step=2, value=5)

    if st.button("Apply"):
        if option == "Smoothing":
            processed_image = apply_smoothing(image, k)
            st.image(processed_image, caption="Smoothed Image", use_column_width=True)
        elif option == "Log Transformation":
            processed_image = apply_log_transformation(image)
            st.image(processed_image, caption="Log Transformed Image", use_column_width=True)
        elif option == "Histogram Equalization":
            processed_image = apply_histogram_equalization(image)
            st.image(processed_image, caption="Histogram Equalized Image", use_column_width=True, channels="GRAY")
        elif option == "Edge Detection":
            processed_image = apply_edge_detection(image)
            st.image(processed_image, caption="Edge Detected Image", use_column_width=True, channels="GRAY")
        elif option == "Sharpening":
            processed_image = apply_sharpening(image)
            st.image(processed_image, caption="Sharpened Image", use_column_width=True)
