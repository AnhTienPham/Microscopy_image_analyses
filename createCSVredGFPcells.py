## This python script creates an csv file for all combined tiff images (2 pages)
## The CSV will contains the number of red nuclei cells, and
## the number of GFP overlapped with the red cells in a given microscopy image
## Instruction: Go to command prompt and change directory to the one containing this file
## Then type "python createCSVredGFPcells.py <folder directory that contains the combined tiff files>"
## For example: cd C:\Users\tiena\OneDrive\Documents\Java,Pymol,Rproj\ImageAnalysis
##              python test.py C:/Users/tiena/OneDrive/Documents/tiff


import sys
import csv
import numpy as np
from scipy import ndimage as ndi
import cv2
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
import os
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from PIL import Image

# Set up the path where your chosen image is located
directory = str(sys.argv[1])
os.chdir(directory)

# Ask which image is which as user prompts
nuclei_reporter = int(input("Which image has the nuclei labeled? (Enter 1 or 2): "))-1
infection_reporter = int(input("Which image is the infection reporter? (Enter 1 or 2): "))-1

##  A method to count the overlapping cells for each single image
def countCells(combinedimage):
    # Split the cells into 2 separate tiff images
    img = Image.open(combinedimage)
    frames = []
    for i in range(2):
        img.seek(i)
        frames.append(img.copy())
    # Choose your mCheery cells
    image = img_as_ubyte(frames[nuclei_reporter])
    print(image)
    # Choose your GFP cells
    seg1 = img_as_ubyte(frames[infection_reporter])

    # Convert the images into a grayscale image
    grey = rgb2gray(image)#color.rgba2rgb(image))
    seg1_grey = rgb2gray(seg1)#color.rgba2rgb(seg1))

    # Setting up the threshold array to identify the cells
    # The class might be modified depending on the cell image, the range is 2 to 4
    thresholds = filters.threshold_multiotsu(grey, classes=2)
    cells = grey > thresholds[0]
    seg1_thresh = filters.threshold_multiotsu(seg1_grey, classes=2)
    seg1_cells = seg1_grey > seg1_thresh[0]

    # Create a distance array to segment the cells from overlapping one
    # The minimum distance might be modified depending on the cell image, the range is 21 to 23
    distance = ndi.distance_transform_edt(cells)
    local_max_coords = feature.peak_local_max(distance, min_distance=23)
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
    masked_local_max_coords = feature.peak_local_max(masked_distance, min_distance=8)
    masked_local_max_mask = np.zeros(masked_distance.shape, dtype=bool)
    masked_local_max_mask[tuple(masked_local_max_coords.T)] = True
    masked_markers = measure.label(masked_local_max_mask)
    masked_segmented_cells = segmentation.watershed(-masked_distance, masked_markers, mask=masked_image)
    # Create a data array that contains the number of red and overlapped cells
    data = []
    data.append(segmented_cells.max())
    data.append(masked_segmented_cells.max())
    return data

## A method to create the CSV file
def createCSV():
    # Create the csv file and the name the columns
    with open('redGFPcells.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image Name", "Number of red nuclei cells", "Number of overlapping red-green cells", "Percentage overlap"])
        # Choose only tiff file to be included
        for filename in os.listdir(directory):
            if filename.endswith(".tiff"):
                getdata = countCells(filename)
                writer.writerow([str(filename), getdata[0], getdata[1], getdata[1]/getdata[0]*100])
            if filename.endswith(".tif"):
                getdata = countCells(filename)
                writer.writerow([str(filename), getdata[0], getdata[1], getdata[1]/getdata[0]*100])

## Creating the CSV file
createCSV()