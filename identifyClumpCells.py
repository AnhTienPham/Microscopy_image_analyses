## This python script counts the number of cells in a given microscopy image
## Depending on the number and size of cells, the minimum distance and class paramters
## may need to be modified

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
from PIL import Image, ImageEnhance

# Set up the path where your chosen image is located
os.getcwd()
os.chdir("/Users/tiena/OneDrive/Documents/tiff/test2")
os.getcwd()

img = Image.open('600uL_1004.tif')
frames = []
for i in range(2):
    img.seek(i)
    frames.append(img.copy())

# Convert the image into a grayscale image
#image = img_as_ubyte(cv2.imread('Cropped2_mCherry_138.tiff'))
image = img_as_ubyte(frames[1])
grey = rgb2gray(image)


# Setting up the threshold array to identify the cells
# The class might be modified depending on the cell image, the range is 2 to 4
thresholds = filters.threshold_multiotsu(grey, classes=2)
cells = grey > thresholds[0]

# Create a distance array to segment the cells from overlapping one
# The minimum distance might be modified depending on the cell image, the range is 5 to 6
distance = ndi.distance_transform_edt(cells)
local_max_coords = feature.peak_local_max(distance, min_distance=7)
local_max_mask = np.zeros(distance.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = measure.label(local_max_mask)

# The cells are segmented and watershed for counting
segmented_cells = segmentation.watershed(-distance, markers, mask=cells)

# Create a 4 subplot figures that show the images transformed and the number of cells
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 5))
fig.suptitle('Cropped_2_mCherry_138')
ax[0,0].imshow(image, cmap='gray')
ax[0,0].set_title('True number of cells: 138')
ax[0,0].axis('off')
ax[0,1].imshow(grey, cmap='gray')
ax[0,1].set_title('Greyscale transformation')
ax[0,1].axis('off')
ax[1,0].imshow(cells, cmap='gray')
ax[1,0].set_title('Identify the cells after thresholding')
ax[1,0].axis('off')
ax[1,1].imshow(color.label2rgb(segmented_cells, bg_label=0))
ax[1,1].set_title('Watershed number of cells: ' + str(segmented_cells.max()))
ax[1,1].axis('off')
plt.show()
