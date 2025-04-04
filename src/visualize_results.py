import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from PIL import Image, ImageOps
import os

def visualize_vertical_displacement(dem_path, csv_path, displacement_csv_path, results_path, base_name):
    """
    Visualize cracks in the DEM and display the calculated vertical displacement for each crack.
    - Show cracks as outlined regions.
    - Display vertical displacement values for each crack.
    """
    os.makedirs(results_path, exist_ok=True)

    if not os.path.exists(dem_path):
        print(f"Error: DEM file not found: {dem_path}")
        return

    dem_img = Image.open(dem_path).convert("L")
    dem_array = np.array(dem_img)

    if not os.path.exists(csv_path):
        print(f"Error: CSV mask file not found: {csv_path}")
        return

    csv_data = pd.read_csv(csv_path, header=None).values

    if not os.path.exists(displacement_csv_path):
        print(f"Error: Displacement CSV file not found: {displacement_csv_path}")
        return

    displacement_df = pd.read_csv(displacement_csv_path)

    labeled_array, num_features = label(csv_data)

    plt.figure(figsize=(10, 8))
    plt.imshow(dem_array, cmap="gray", interpolation="none")

    for crack_label in range(1, num_features + 1):
        crack_mask = (labeled_array == crack_label)
        crack_positions = np.column_stack(np.where(crack_mask))
        plt.scatter(crack_positions[:, 1], crack_positions[:, 0], label=f"Crack {crack_label}", s=5)

        displacement_row = displacement_df[displacement_df['crack_label'] == crack_label]

        if not displacement_row.empty:
            displacement = displacement_row['vertical_displacement'].values[0]
            centroid_y = np.mean(crack_positions[:, 0])
            centroid_x = np.mean(crack_positions[:, 1])

            plt.text(centroid_x, centroid_y, f"{displacement:.3f}m", color='red', fontsize=12, ha='center')

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.legend().set_visible(False)

    # Save temporary image
    temp_path = os.path.join(results_path, f"{base_name}_temp.png")
    plt.savefig(temp_path, dpi=100, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Load the saved image and process it to 256x256 without white borders
    temp_img = Image.open(temp_path)
    temp_img = ImageOps.crop(temp_img)  # Remove excess borders
    temp_img = temp_img.resize((256, 256), Image.Resampling.LANCZOS)

    # Save the final image
    final_path = os.path.join(os.path.join(results_path, "labeled_rgb"), f"{base_name}VERT_DISP.png")
    temp_img.save(final_path)
    os.remove(temp_path)  # Clean up temp file

    print(f"Visualization saved: {final_path}")

def visualize_looping(dem_folder, csv_folder, displacement_folder, results_folder):
    """Loops through all displacement CSV files and visualizes vertical displacement."""
    
    # Ensure the results folder exists
    os.makedirs(results_folder, exist_ok=True)

    # Loop through all displacement CSV files
    for disp_filename in sorted(os.listdir(displacement_folder)):
        if disp_filename.endswith("VERT_DISP.csv"):  # Adjust file extension if needed
            displacement_csv_path = os.path.join(displacement_folder, disp_filename)

            # Generate corresponding file paths
            base_name = disp_filename.replace("VERT_DISP.csv", "")  # Extract base name
            dem_path = os.path.join(dem_folder, base_name + "RGB.jpg")
            csv_path = os.path.join(csv_folder, base_name + "MASK.csv")

            # Visualize vertical displacement
            visualize_vertical_displacement(dem_path, csv_path, displacement_csv_path, results_folder, base_name)

            print(f"Processed: {disp_filename} → Visualization saved")