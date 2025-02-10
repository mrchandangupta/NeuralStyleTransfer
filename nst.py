import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Function to deprocess the image
def deprocess_image(img):
    img = img.reshape((224, 224, 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Function to compute content loss
def compute_content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

# Function to compute gram matrix for style representation
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Function to compute style loss
def compute_style_loss(style, generated):
    style_gram = gram_matrix(style)
    generated_gram = gram_matrix(generated)
    return tf.reduce_mean(tf.square(style_gram - generated_gram))

# Function to compute total loss
def compute_loss(content_features, style_features, generated_features, content_weight, style_weight):
    content_loss = compute_content_loss(content_features[0], generated_features[0])
    style_loss = sum([compute_style_loss(style, gen) for style, gen in zip(style_features, generated_features[1:])])
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss

# Load VGG19 model
def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_layers = ['block5_conv2']
    # Adding more layers for style representation
    style_layers = [
        'block1_conv1', 'block1_conv2', 'block2_conv1', 
        'block2_conv2', 'block3_conv1', 'block3_conv2', 
        'block4_conv1', 'block4_conv2', 'block5_conv1', 
        'block5_conv2'
    ]
    outputs = [vgg.get_layer(name).output for name in (content_layers + style_layers)]
    model = tf.keras.Model([vgg.input], outputs)
    return model, content_layers, style_layers

# Run style transfer
def run_style_transfer(content_image, style_image, num_iterations=200, content_weight=1e3, style_weight=1.15e-1, target_size=(224, 224)):
    content_image = preprocess_image(content_image, target_size)
    style_image = preprocess_image(style_image, target_size)
    generated_image = tf.Variable(content_image, dtype=tf.float32)

    model, content_layers, style_layers = get_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.08)  # Adjust the learning rate if needed
    # You can also try other optimizers like SGD or RMSprop
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.08)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.08)

    # Extract features
    content_targets = model(content_image)[:1]
    style_targets = model(style_image)[1:]

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            generated_outputs = model(generated_image)
            generated_content = generated_outputs[:1]
            generated_styles = generated_outputs[1:]
            loss = compute_loss(content_targets, style_targets, [generated_content] + generated_styles, content_weight, style_weight)

        gradients = tape.gradient(loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        generated_image.assign(tf.clip_by_value(generated_image, -128.0, 128.0))

    final_image = deprocess_image(generated_image.numpy())
    return final_image

# Streamlit app
st.title("Neural Style Transfer")
st.markdown("##### Elevate your artistry and transform your images into breathtaking masterpieces with our neural style transfer tool. Effortlessly blend your unique creations with renowned art styles, niches, and genres, giving your art a fresh, new perspective.")
# Display demonstration image 

# Header for the previous arts section
st.text(" Here is our previous arts")

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
    content_image = Image.open(content_image_file)
    style_image = Image.open(style_image_file)

    st.image(content_image, caption="Content Image", use_container_width=True)
    st.image(style_image, caption="Style Image", use_container_width=True)

    if st.button("Apply Style Transfer"):
        with st.spinner("Processing..."):
            try:
                styled_image = run_style_transfer(content_image, style_image, num_iterations=200, target_size=(224, 224))
                st.image(styled_image, caption="Styled Image", use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.write("Please upload both a content image and a style image.")