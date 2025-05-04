import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import label

def compute_vertical_displacement(predicted_path, dem_path, csv_path, output_csv):
    """
    Calculate vertical displacement for each crack using normalized elevation values.
    - Find left and right edges of the crack.
    - Compute vertical displacement as the difference between max(right) - min(left).
    """
    # Load predicted segmentation (binary mask, where 1 represents a joint)
    predicted_img = Image.open(predicted_path).convert("L")
    predicted_array = np.array(predicted_img)

    # Load DEM elevation image (grayscale, where the pixel values represent heights)
    dem_img = Image.open(dem_path).convert("L")
    dem_array = np.array(dem_img)

    # Normalize DEM elevation values (convert pixel values to real-world meters)
    min_elevation = 0  # Minimum real-world elevation (meters)
    max_elevation = 0.0254  # Maximum real-world elevation (meters, 1 inches)
    elevation_data = (dem_array / 255) * (max_elevation - min_elevation) + min_elevation

    # Load CSV mask (binary joint mask, 1 = joint, 0 = background)
    csv_data = pd.read_csv(csv_path, header=None).values  # Load as NumPy array

    # Ensure all arrays have the same shape
    if predicted_array.shape != elevation_data.shape or predicted_array.shape != csv_data.shape:
        raise ValueError("Image and CSV dimensions do not match!")

    # Use connected component labeling to group adjacent 1s into cracks
    labeled_array, num_features = label(csv_data)  # Find connected regions of 1s (cracks)

    # List to store vertical displacement for each crack
    displacements = []

    # Iterate over each unique labeled crack region
    for crack_label in range(1, num_features + 1):
        # Get the positions of this crack (all pixels with the current label)
        crack_mask = (labeled_array == crack_label)
        crack_positions = np.column_stack(np.where(crack_mask))

        # Find the left edge (closest 0 to the left of the crack)
        left_heights = []
        for y, x in crack_positions:
            left_x = x - 1
            while left_x >= 0 and csv_data[y, left_x] == 1:
                left_x -= 1
            if left_x >= 0:  # Valid left edge
                left_heights.append(elevation_data[y, left_x])

        # Find the right edge (closest 0 to the right of the crack)
        right_heights = []
        for y, x in crack_positions:
            right_x = x + 1
            while right_x < elevation_data.shape[1] and csv_data[y, right_x] == 1:
                right_x += 1
            if right_x < elevation_data.shape[1]:  # Valid right edge
                right_heights.append(elevation_data[y, right_x])

        # Ensure valid left and right edges were found
        if len(left_heights) == 0 or len(right_heights) == 0:
            continue  # Skip if no valid edge found

        # Get the min height for the left side and max height for the right side
        min_left_height = np.min(left_heights)
        max_right_height = np.max(right_heights)

        # Compute vertical displacement: max(right) - min(left)
        vertical_displacement = max_right_height - min_left_height

        # Store result for this crack
        displacements.append([crack_label, vertical_displacement])

    # Convert to DataFrame and save results
    df_displacements = pd.DataFrame(displacements, columns=["crack_label", "vertical_displacement"])
    df_displacements.to_csv(output_csv, index=False)

    print(f"Processed {len(displacements)} cracks.")
    print(f"Saved vertical displacement data to {output_csv}")


def vertical_displacement_looping(seg_folder, dem_folder, csv_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all segmentation images
    for seg_filename in sorted(os.listdir(seg_folder)):
        if seg_filename.endswith("SEG.jpg"):
            seg_path = os.path.join(seg_folder, seg_filename)

            # Generate corresponding file paths
            dem_path = os.path.join(dem_folder, seg_filename.replace("SEG.jpg", "DEM.jpg"))
            csv_path = os.path.join(csv_folder, seg_filename.replace("SEG.jpg", "MASK.csv"))
            output_csv = os.path.join(output_folder, seg_filename.replace("SEG.jpg", "VERT_DISP.csv"))

            # Compute vertical displacement
            compute_vertical_displacement(seg_path, dem_path, csv_path, output_csv)

            print(f"Processed: {seg_filename} → {output_csv}")