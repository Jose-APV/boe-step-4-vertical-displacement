import os
import numpy as np
from PIL import Image
import math
from natsort import natsorted  # Ensures correct numeric sorting

def reassemble_image(tiles_folder, output_folder, original_width, original_height, tile_size=256):
    """
    Reassembles 256x256 tiles into the full sidewalk orthoimage.
    This algorithm will take the labeled_rgb_path which is the folder path that contains all the rgb images that have been 
    labeled with measurements. This path is in the results folder.
    The original width and height of the original path is automatically passed in.
    Tile size is always 256 because labeled images are 256x256
    """
    # Get all tile filenames matching the prefix
    tile_filenames = [f for f in os.listdir(tiles_folder) if f.endswith(('.png', '.jpg'))]
    tile_filenames = natsorted(tile_filenames)  # Natural sorting to avoid issues

    # Compute number of rows & columns, rounding up to include edge patches
    num_cols = math.ceil(original_width / tile_size)
    num_rows = math.ceil(original_height / tile_size)

    print(f"Reassembling image with {num_rows} rows and {num_cols} columns...")

    # Validate that we have the correct number of tiles
    if len(tile_filenames) < num_rows * num_cols:
        print(f"Warning: Expected {num_rows * num_cols} tiles but found {len(tile_filenames)}!")
    
    # List to store image rows
    image_rows = []

    for row in range(num_rows):
        row_tiles = []
        for col in range(num_cols):
            tile_index = row * num_cols + col  # Compute index from sorted list
            
            if tile_index >= len(tile_filenames):
                print(f"Missing tile at row {row}, col {col} (index {tile_index})")
                continue  # Avoid out-of-bounds errors

            tile_path = os.path.join(tiles_folder, tile_filenames[tile_index])  # Get file path
            
            # Load tile safely
            try:
                tile_img = Image.open(tile_path)
            except Exception as e:
                print(f"Error loading {tile_path}: {e}")
                continue

            row_tiles.append(np.array(tile_img))  # Convert to NumPy array

        if row_tiles:
            # Stack images horizontally to form a row
            image_rows.append(np.hstack(row_tiles))

    if not image_rows:
        print("Error: No image rows were created. Check tile ordering.")
        return

    # Stack rows vertically to form the final image
    full_image = np.vstack(image_rows)

    # Convert to a PIL image
    final_image = Image.fromarray(full_image)

    # Automatically generate a file name for the output if not provided
    if not output_folder.lower().endswith(('.png', '.jpg', '.jpeg')):
        output_folder = os.path.join(output_folder, 'recreated_image.png')  # Default to 'recreated_image.png'

    final_image.save(output_folder)

    print(f"Reassembled image saved to {output_folder}")





