import matplotlib.pyplot as plt
from PIL import Image
import os

from vertical_displacement import compute_vertical_displacement, vertical_displacement_looping
from visualize_results import visualize_looping, visualize_vertical_displacement
from segmentation2binarymask import convert_all_masks, image_to_csv
from unet import process_segmentation, test_model
from resize_dem_and_ortho import split_dem_image
from resize_dem_and_ortho import split_testing_images
from reassemble_labeledRGB_images import reassemble_image


def main(sidewalk_name):
    # Necessary Paths
    # change this to the desired sidewalk
    sidewalk_path = "/Users/jose/pointcloud_files/Demo/" + sidewalk_name # make your sidewalk structure similar to this

    # Don't change this
    # Define the results folder path
    results_path = os.path.join(sidewalk_path, "results")
    os.makedirs(results_path, exist_ok=True)
    labeled_rgb_with_measurements_path = os.path.join(results_path, "labeled_rgb") # a folder path containing all the cut RGB pictures with elevation measurements edited
    os.makedirs(labeled_rgb_with_measurements_path, exist_ok=True)

    original_dem_path = sidewalk_path + "/" + sidewalk_name +"DEM.jpg"
    original_RGB_path = sidewalk_path + "/" + sidewalk_name +"RGB.jpg"  # Path to the image you want to test

    sidewalk_output_folder_rgb = sidewalk_path + "/resized_rgb/" # for rgb
    sidewalk_output_folder_dem = sidewalk_path + "/resized_dem/" # for dem

    pretrained_model_path = '../src/unet_membrane.hdf5'

    # loop through rgb and dem images, name them by 12345
    resized_rgb_path = sidewalk_output_folder_rgb
    resized_dem_path = sidewalk_output_folder_dem # Don't enter original DEM path

    # also this, need to save the predicted image by numbers, predicted_{counter} + png
    predicted_seg_label_path = sidewalk_path + "/labeled_prediction/" # Need better naming for this


    # also this, need multiple csv files, naming should follow 12345
    binary_mask_csv_path = results_path  # Binary mask CSV

    # same here, naming follow 12345
    vertical_displacement_csv = results_path # displacement table aka results but in a table
    img_size = 256  # Set the image size to match the model input


    # split the DEM images
    # You input the DEM or Elevation image, and begins the process of splitting the images. Places in original sidewalk path
    split_dem_image(original_dem_path, sidewalk_output_folder_dem)

    # splits the RGB value
    split_testing_images(original_RGB_path, sidewalk_output_folder_rgb)

  
    # Have the u-net pre-trained model predict the labeled segmentation
    process_segmentation(pretrained_model_path, sidewalk_output_folder_rgb, img_size, predicted_seg_label_path)

    # Convert image to csv binary mask
    convert_all_masks(predicted_seg_label_path, binary_mask_csv_path)

    # Actually calculates the displacement in the areas that have segments
    vertical_displacement_looping(predicted_seg_label_path, resized_dem_path, binary_mask_csv_path, vertical_displacement_csv)

    # Visualize where the displacement occurs and mark displacement height
    # should visualize all images and stick them togther
    visualize_looping(resized_rgb_path, binary_mask_csv_path, vertical_displacement_csv, results_path)

    reassemble_image(labeled_rgb_with_measurements_path, results_path)
    
if __name__ == "__main__":
    base_path = "/Users/jose/pointcloud_files/Demo/"

    # Get all folder names inside base_path (only directories)
    all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    # Loop through each sidewalk folder and call main with just the folder name
    for folder in all_folders:
        print(f"Found sidewalk: {folder}")  # Debugging output
        main(folder)







# I need to first split the images, DEM and RGB. Therefore each DEM will have a dedicated RGB 
# Then we need to loop through each RGB image and predict the segmentation for each RGB cut image, and save somewere easy to retrieve
# then, for each RGB image, we need to make a csv binary mask file with appropiate names.
# then, for each dem image, we will need to loop through each dem&predicted labels and glue and compute a vertical displacement 
# Then we need to overlay the image and save it wwhile looping through each predicted dem
# then we need to glue visualization image together


# Notes, as of now, resized dems are stored in the original folder while resized RGB have their own folder. Consider moving everything to parent folder