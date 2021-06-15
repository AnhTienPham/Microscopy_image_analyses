## This python script counts the number of GFP cells
# that is overlapped with the red cells in a given microscopy image

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import cv2
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
import os
from skimage import img_as_ubyte
from skimage.color import rgb2gray

# Set up the path where your chosen image is located
os.getcwd()
os.chdir("/Users/tiena/OneDrive/Documents/Java,Pymol,Rproj/ImageAnalysis")
os.getcwd()

# Choose your mCheery cells
image = img_as_ubyte(cv2.imread('Cropped2_mCherry_138.tiff'))
# Choose your GFP cells
seg1 = img_as_ubyte(cv2.imread('Cropped2_GFP.tiff'))

# Convert the images into a grayscale image
grey = rgb2gray(image)
seg1_grey = rgb2gray(seg1)

# Setting up the threshold array to identify the cells
# The class might be modified depending on the cell image, the range is 2 to 4
thresholds = filters.threshold_multiotsu(grey, classes=2)
cells = grey > thresholds[0]
seg1_thresh = filters.threshold_multiotsu(seg1_grey, classes=2)
seg1_cells = seg1_grey > seg1_thresh[0]

# Create a distance array to segment the cells from overlapping one
# The minimum distance might be modified depending on the cell image, the range is 5 to 6
distance = ndi.distance_transform_edt(cells)
local_max_coords = feature.peak_local_max(distance, min_distance=5)
local_max_mask = np.zeros(distance.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = measure.label(local_max_mask)

# The cells are segmented and watershed for counting
segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

# Mask the two images together
masked_image = segmented_cells * seg1_cells

# Remove small objects, the higher the unit of morphology, the larger the objects removed
morp = morphology.disk(3)
small_obj = morphology.white_tophat(masked_image, morp)
masked_image = masked_image - small_obj

# Do the same process for the new overlap image as above
masked_distance = ndi.distance_transform_edt(masked_image)
masked_local_max_coords = feature.peak_local_max(masked_distance, min_distance=5)
masked_local_max_mask = np.zeros(masked_distance.shape, dtype=bool)
masked_local_max_mask[tuple(masked_local_max_coords.T)] = True
masked_markers = measure.label(masked_local_max_mask)
masked_segmented_cells = segmentation.watershed(-masked_distance, masked_markers, mask=masked_image)

# Create a 4 subplot figures that show the images transformed and the number of cells
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 5))
fig.suptitle('Cropped_2_mCherry_138')
ax[0,0].imshow(cells, cmap='gray')
ax[0,0].set_title('True number of red nuclei cells: 138')
ax[0,0].axis('off')
ax[0,1].imshow(seg1_cells, cmap='gray')
ax[0,1].set_title('GFP cells')
ax[0,1].axis('off')
ax[1,0].imshow(masked_image, cmap='gray')
ax[1,0].set_title('Overlapping cells')
ax[1,0].axis('off')

# Show the final watershed cells from the overlapping of the two images
ax[1,1].imshow(color.label2rgb(masked_segmented_cells, bg_label=0))
# Count the number of cells
ax[1,1].set_title('Watershed number of cells: ' + str(masked_segmented_cells.max()))
ax[1,1].axis('off')
plt.show()
