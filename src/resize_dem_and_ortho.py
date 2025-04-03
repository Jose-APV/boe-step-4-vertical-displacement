import os
from PIL import Image


def split_dem_image(input_image_path, patch_size=256):
    """Splits a single DEM image into 256x256 patches and stores them in the same folder."""
    
    # Get the directory of the input image and create the output folder
    output_folder = os.path.dirname(input_image_path)
    os.makedirs(os.path.dirname(input_image_path), exist_ok=True) 

    # Open the DEM image
    with Image.open(input_image_path) as dem_img:
        width, height = dem_img.size
        patch_idx = 1  # Patch numbering

        # Loop through the image and extract patches
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                # Crop DEM image to create a patch
                dem_patch = dem_img.crop((x, y, x + patch_size, y + patch_size))

                # Save patches with format {PatchIndex}DEM.jpg (e.g., "1DEM.jpg")
                output_path = os.path.join(output_folder, f"{patch_idx}DEM.jpg")
                dem_patch.save(output_path)

                print(f"Saved: {output_path}")
                patch_idx += 1





def split_testing_images(input_folder, output_folder, patch_size=256):  # Changed patch_size to 256
    # Create output folder for patches
    os.makedirs(output_folder, exist_ok=True)

    patch_idx = 0  # Start numbering patches

    # Get only images matching the testing pattern (e.g., "sidewalk_1RGB.jpg")
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith("RGB.jpg")])

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Loop through the image and extract 256x256 patches
                for y in range(0, height, patch_size):
                    for x in range(0, width, patch_size):
                        patch = img.crop((x, y, x + patch_size, y + patch_size))
                        
                        # Ensure patches are exactly 256x256 (handle edge cases)
                        if patch.size != (patch_size, patch_size):
                            patch = patch.resize((patch_size, patch_size), Image.LANCZOS)
                        
                        # Save patches with continuous numbering
                        output_path = os.path.join(output_folder, f"{patch_idx}.png")
                        patch.save(output_path)
                        print(f"Saved: {output_path}")
                        patch_idx += 1
                        
        except Exception as e:
            print(f"Error processing {filename}: {e}")


