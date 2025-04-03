import matplotlib.pyplot as plt
from PIL import Image
import os

from vertical_displacement import compute_vertical_displacement
from visualize_results import visualize_vertical_displacement
from segmentation2binarymask import image_to_csv
from unet import test_model
from resize_dem_and_ortho import split_dem_image
from resize_dem_and_ortho import split_testing_images


def display_predicted_image(resized_rgb_path, predicted_seg_label_path):
    img = Image.open(resized_rgb_path)
    output_img = Image.open(predicted_seg_label_path)
    output_img = output_img.convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray")  # Show input image
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(output_img, cmap="gray")  # Show predicted image
    axes[1].set_title("Predicted Output")
    axes[1].axis('off')

    plt.show()


def main():
    # Necessary Paths
    sidewalk_name = "sidewalk_37" # change this to the desired sidewalk
    sidewalk_path = "/Users/jose/pointcloud_files/Demo/" + sidewalk_name # make your sidewalk structure similar to this

    # Don't change this
    # Define the results folder path
    results_path = os.path.join(sidewalk_path, "results")
    os.makedirs(results_path, exist_ok=True)

    original_dem_path = sidewalk_path + "/" + sidewalk_name +"DEM.jpg"
    original_RGB_path = sidewalk_path + "/" + sidewalk_name +"RGB.jpg"  # Path to the image you want to test
    sidewalk_input_folder = sidewalk_path # for rgb
    sidewalk_output_folder = sidewalk_path + "/resized_rgb/" # for rgb
    resized_rgb_path = sidewalk_output_folder + "/0.png"
    resized_dem_path = sidewalk_path + "/1DEM.jpg" # Don't enter original DEM path
    pretrained_model_path = '../unet_membrane.hdf5'
    predicted_seg_label_path = sidewalk_path + "/labeled_prediction/"+ "predicted_0"+".png" # Need better naming for this
    binary_mask_csv_path = results_path + "/output_data.csv"  # Binary mask CSV
    vertical_displacement_csv = results_path + "/vertical_displacement.csv" # displacement table aka results but in a table
    img_size = 256  # Set the image size to match the model input


    # split the DEM images
    # You input the DEM or Elevation image, and begins the process of splitting the images. Places in original sidewalk path
    split_dem_image(original_dem_path)

    # splits the RGB value
    split_testing_images(sidewalk_input_folder, sidewalk_output_folder)

    # Have the u-net pre-trained model predict the labeled segmentation
    test_model(pretrained_model_path, resized_rgb_path, img_size, predicted_seg_label_path) #changed this lol

    display_predicted_image(resized_rgb_path, predicted_seg_label_path)

    # Convert image to csv binary mask
    image_to_csv(predicted_seg_label_path, binary_mask_csv_path)

    # Actually calculates the displacement in the areas that have segments
    compute_vertical_displacement(predicted_seg_label_path, resized_dem_path, binary_mask_csv_path, vertical_displacement_csv)

    # Visualize where the displacement occurs and mark displacement height
    visualize_vertical_displacement(resized_rgb_path, binary_mask_csv_path, vertical_displacement_csv, results_path)
    
if __name__ == "__main__":
    main()







# I need to first split the images, DEM and RGB. Therefore each DEM will have a dedicated RGB 
# Then we need to loop through each RGB image and predict the segmentation for each RGB cut image, and save somewere easy to retrieve
# then, for each RGB image, we need to make a csv binary mask file with appropiate names.
# then, for each dem image, we will need to loop through each dem&predicted labels and compute a vertical displacement 
# Then we need to overlay the image and save it while looping through each predicted dem
# then we need to glue visualization image together


# Notes, as of now, resized dems are stored in the original folder while resized RGB have their own folder. Consider moving everything to parent folder