import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from PIL import Image
import os

def visualize_vertical_displacement(dem_path, csv_path, displacement_csv_path, results_path):
    """
    Visualize cracks in the DEM and display the calculated vertical displacement for each crack.
    - Show cracks as outlined regions.
    - Display vertical displacement values for each crack.
    """
    # Load DEM elevation image (grayscale, where pixel values represent heights)
    dem_img = Image.open(dem_path).convert("L")
    dem_array = np.array(dem_img)

    # Load CSV mask (binary joint mask, 1 = joint, 0 = background)
    csv_data = pd.read_csv(csv_path, header=None).values  # Load as NumPy array

    # Load vertical displacement results (crack_label, vertical_displacement)
    displacement_df = pd.read_csv(displacement_csv_path)

    # Step 1: Use connected component labeling to group adjacent 1s into cracks
    labeled_array, num_features = label(csv_data)  # Find connected regions of 1s (cracks)

    # Create a figure to plot
    plt.figure(figsize=(10, 8))

    # Show DEM image as background
    plt.imshow(dem_array, cmap="gray", interpolation="none")

    # Iterate over each unique labeled crack region
    for crack_label in range(1, num_features + 1):
        # Get the positions of this crack (all pixels with the current label)
        crack_mask = (labeled_array == crack_label)
        crack_positions = np.column_stack(np.where(crack_mask))

        # Plot the crack boundary (outline of the crack region)
        plt.scatter(crack_positions[:, 1], crack_positions[:, 0], label=f"Crack {crack_label}", s=5)

        # Get the vertical displacement for this crack
        displacement = displacement_df[displacement_df['crack_label'] == crack_label]['vertical_displacement'].values[0]

        # Annotate the displacement near the crack's centroid
        centroid_y = np.mean(crack_positions[:, 0])
        centroid_x = np.mean(crack_positions[:, 1])

        plt.text(centroid_x, centroid_y, f"{displacement:.3f}m", color='red', fontsize=12, ha='center')


    plt.title("Vertical Displacement for Each Crack", fontsize=16)
    plt.xlabel("X-coordinate", fontsize=12)
    plt.ylabel("Y-coordinate", fontsize=12)
    plt.legend(loc='upper right')
    plt.colorbar(label="Elevation (Height)")
    output_image_path = os.path.join(results_path, "vertical_displacement_image.png") # create image name
    plt.savefig(output_image_path, bbox_inches='tight') # save image into results folder
    plt.show()
    
