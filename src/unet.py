import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from PIL import Image
import os

def load_data_testing(img_path, img_size):
    """Load and preprocess a single image for prediction."""
    try:
        img = load_img(img_path, target_size=(img_size, img_size), color_mode='grayscale')  # Load as grayscale
        img_array = img_to_array(img) / 255.0  # Normalize the image (0 to 1)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array, img  # Return both the array and original image
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None, None

def test_model(model_path, img_path, img_size, output_path):
    """Load the model, test on an input image, and visualize the result."""
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the input image
    img_array, original_img = load_data_testing(img_path, img_size)
    if img_array is None:
        return

    # Make prediction
    prediction = model.predict(img_array)

    # Convert prediction to a binary mask
    pred_mask = np.squeeze(prediction)  # Remove batch dimension
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Threshold to 0 or 255

    # Convert to image and save
    predicted_img = Image.fromarray(pred_mask, mode="L")  # Ensure grayscale format
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    predicted_img.save(output_path)
    print(f"Prediction image saved at: {output_path}")
