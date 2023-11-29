# app.py
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pickle

# Load the pre-trained model
model = load_model('C:/Users/ASUS/Downloads/Model/object_classification_model.h5')
with open('C:/Users/ASUS/Downloads/Model/label.pkl', 'rb') as file:
    labels = pickle.load(file)

# Define your class labels
# class_labels = ['Bacterialblight', 'Blast', 'Brownspot','Tungro']

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))  # Update with your model's target image size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Streamlit app
st.title("CIFAR-10 Object Detection")

# File uploader
uploaded_file = st.file_uploader("Choose a image", type=["jpg", "jpeg", "png"])

# Display the uploaded image and make predictions
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Check if the file uploaded is an image
    if uploaded_file.type.startswith('image'):
        # Preprocess the uploaded image
        img_array = preprocess_image(uploaded_file)

        # Make a prediction
        prediction = model.predict(img_array)
        
        # Get the predicted class label
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = [x for x in labels if labels[x] == predicted_class_index][0]

        # Display the prediction result
        st.write("Prediction Result:")
        st.write(predicted_class_label)
