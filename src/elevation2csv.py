import os
import numpy as np
from PIL import Image
import csv

def dem_to_csv(dem_path, output_csv, min_elevation=0.0, max_elevation=0.0254):
    """Convert a DEM grayscale image to a CSV file containing elevation values in meters."""
    dem_img = Image.open(dem_path).convert("L")
    dem_array = np.array(dem_img)
    
    elevation_data = (dem_array / 255.0) * (max_elevation - min_elevation) + min_elevation

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(elevation_data)

def convert_all_dem_images(dem_folder, output_folder):
    """Convert all DEM images in a folder to CSV elevation files."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(dem_folder)):
        if filename.endswith("DEM.jpg"):
            dem_path = os.path.join(dem_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_csv = os.path.join(output_folder, f"{base_name}.csv")

            dem_to_csv(dem_path, output_csv)
