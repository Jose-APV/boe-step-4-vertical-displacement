import cv2
import numpy as np

# load gray elevation
# gray_image = cv2.imread("/Users/jose/Downloads/2DEM copy.jpg", cv2.IMREAD_GRAYSCALE)
gray_image = cv2.imread("/Users/jose/pointcloud_files/Demo/sidewalk_12/sidewalk_12DEM.jpg", cv2.IMREAD_GRAYSCALE)

# Apply a colormap, u can choose from JET, VIRIDIS, or PLASMA
colored_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

cv2.imwrite("colored_elevation.png", colored_image)
cv2.imshow("Colored Elevation", colored_image)
cv2.waitKey(0) # press key to leave
cv2.destroyAllWindows()
