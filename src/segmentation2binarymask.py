import numpy as np
from PIL import Image
import csv

def image_to_csv(image_path, output_csv):
    """
    Convert a grayscale image to a CSV file where:
    - 0 represents the white background
    - 1 represents the black sidewalk joints
    """
    # Load the image in grayscale mode
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Convert white (255) to 0 and black (0) to 1
    binary_array = np.where(img_array < 128, 1, 0)  # Thresholding (anything darker than 128 -> 1, else 0)
    
    # Save as CSV
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in binary_array:
            writer.writerow(row)
    
    print(f"CSV file saved: {output_csv}")