import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Load the trained model
#model = load_model('Plant Disease CNN2.keras') 
model=load_model('best_model1.h5')
# Define class names (update according to your model's output classes)
class_names = list(ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    'New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
).class_indices.keys())

# Function to load and preprocess the image
def load_and_preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((128,128))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("Plant Disease Detection with CNN Model")
st.write("Upload a JPG image for detection of Plant Disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a JPG file", type=['jpg', 'jpeg','png'])

if uploaded_file is not None:
    # Load and display the image
    image = load_and_preprocess_image(uploaded_file)

    # Make prediction
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the prediction
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption='Uploaded Image')
        with col2:
            st.markdown(f'The model detects this as: **{predicted_class.split("___")[1].replace("_"," ")}**')