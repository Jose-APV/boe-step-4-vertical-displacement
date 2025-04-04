import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from PIL import Image
import os

def visualize_vertical_displacement(dem_path, csv_path, displacement_csv_path, results_path, base_name):
    """
    Visualize cracks in the DEM and display the calculated vertical displacement for each crack.
    - Show cracks as outlined regions.
    - Display vertical displacement values for each crack.
    """
    # Ensure output directory exists
    os.makedirs(results_path, exist_ok=True)

    # Load DEM elevation image (grayscale, where pixel values represent heights)
    if not os.path.exists(dem_path):
        print(f"Error: DEM file not found: {dem_path}")
        return

    dem_img = Image.open(dem_path).convert("L")
    dem_array = np.array(dem_img)

    # Load CSV mask (binary joint mask, 1 = joint, 0 = background)
    if not os.path.exists(csv_path):
        print(f"Error: CSV mask file not found: {csv_path}")
        return

    csv_data = pd.read_csv(csv_path, header=None).values  # Load as NumPy array

    # Load vertical displacement results
    if not os.path.exists(displacement_csv_path):
        print(f"Error: Displacement CSV file not found: {displacement_csv_path}")
        return

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
        displacement_row = displacement_df[displacement_df['crack_label'] == crack_label]

        if not displacement_row.empty:
            displacement = displacement_row['vertical_displacement'].values[0]
            
            # Annotate the displacement near the crack's centroid
            centroid_y = np.mean(crack_positions[:, 0])
            centroid_x = np.mean(crack_positions[:, 1])

            plt.text(centroid_x, centroid_y, f"{displacement:.3f}m", color='red', fontsize=12, ha='center')
    
    
    # Remove ticks, labels, and color bar
    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.xlabel('')  # Remove x-axis label
    plt.ylabel('')  # Remove y-axis label
    plt.legend().set_visible(False)  # Remove legend

    # Save visualization image with unique name
    # we do this before adding title and labels because we want to save the pure image
    # image saved should be 256x256
    plt.gcf().set_size_inches(2.56, 2.56)  # Resize figure
    plt.savefig(f"{results_path}/labeled_rgb/{base_name}VERT_DISP.png", dpi=100, bbox_inches='tight', pad_inches=0) # save image into results/labeled_rgb folder
    output_image_path = os.path.join(results_path, f"{base_name}VERT_DISP.png")  
    plt.title(f"Vertical Displacement - {base_name}", fontsize=16)
    plt.xlabel("X-coordinate", fontsize=12)
    plt.ylabel("Y-coordinate", fontsize=12)
    plt.legend(loc='upper right')
    plt.colorbar(label="Elevation (Height)")

    

    """remove '#' to view the result"""
    #plt.show()
    
    plt.close()  # Close the plot to free memory

    print(f"Visualization saved: {output_image_path}")

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