import os
import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub
import logging
import streamlit as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load TensorFlow Hub module
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

logging.basicConfig(level=logging.INFO)

def load_and_process_image(image_path):
    logging.info(f"Loading and processing image from {image_path}")
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    return tf.convert_to_tensor(img, dtype=tf.float32)[tf.newaxis, ...]

def tensor_to_image(tensor):
    logging.info("Converting tensor to image")
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def perform_style_transfer(content_image_path, style_image_path):
    try:
        logging.info("Performing style transfer")
        content_image = load_and_process_image(content_image_path)
        style_image = load_and_process_image(style_image_path)
        outputs = hub_module(content_image, style_image)
        stylized_image = outputs[0]
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "generated_image.jpg")
        tensor_to_image(stylized_image).save(output_path)
        logging.info(f"Style transfer complete, saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error during style transfer: {e}")
        return None

# Streamlit app
st.title("Neural Style Transfer")
st.markdown("##### Elevate your artistry and transform your images into breathtaking masterpieces with our neural style transfer tool. Effortlessly blend your unique creations with renowned art styles, niches, and genres, giving your art a fresh, new perspective.")

# Header for the previous arts section
st.text("Here is our previous arts")

# Load the images from the local directory
image_path_1 = "demo2.jpeg"
image_path_2 = "demo.jpeg"

# Function to load image from local path
def load_image(path):
    try:
        return Image.open(path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Load images
image1 = load_image(image_path_1)
image2 = load_image(image_path_2)

# Display images side by side
col1, col2 = st.columns(2)

with col1:
    if image1:
        st.image(image1, caption="Previous Art 1", use_container_width=True)

with col2:
    if image2:
        st.image(image2, caption="Previous Art 2", use_container_width=True)

# Upload images
content_image_file = st.file_uploader("Choose a content image", type=["jpg", "jpeg", "png"])
style_image_file = st.file_uploader("Choose a style image", type=["jpg", "jpeg", "png"])

if content_image_file and style_image_file:
    content_image_path = os.path.join("uploads", content_image_file.name)
    style_image_path = os.path.join("uploads", style_image_file.name)
    
    os.makedirs("uploads", exist_ok=True)
    
    with open(content_image_path, "wb") as f:
        f.write(content_image_file.getbuffer())
    
    with open(style_image_path, "wb") as f:
        f.write(style_image_file.getbuffer())
    
    st.image(content_image_file, caption="Content Image", use_container_width=True)
    st.image(style_image_file, caption="Style Image", use_container_width=True)

    if st.button("Apply Style Transfer"):
        with st.spinner("Processing..."):
            output_path = perform_style_transfer(content_image_path, style_image_path)
            if output_path:
                st.image(output_path, caption="Styled Image", use_container_width=True)
            else:
                st.error("Style transfer failed")
else:
    st.write("Please upload both a content image and a style image.")