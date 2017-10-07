import os
from glob import glob
import time

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage import color
from skimage import feature
from skimage import filters

import rpsutil as rps

def generateGrayFeatures(filename='gray.csv', binary=False):
    """Reads image files, generates features from grayscale image and saves the
    features and labels in a csv file."""

    t0 = time.time()
    
    gestures = [rps.ROCK, rps.PAPER, rps.SCISSORS]

    # Create a list of image files for each gesture
    files = []
    for i, gesture in enumerate(gestures):
        path = os.path.join(rps.imgPathsRaw[gesture], '*.png')
        files.append(glob(path))
        files[i].sort(key=str.lower)

    nbImages = sum([len(i) for i in files])

    # Create empty numpy arays for features and labels
    features = np.empty((nbImages, 240*280), dtype=np.float)
    labels = np.empty((nbImages, 2), dtype=np.object)

    # Generate grayscale image
    counter = 0
    for i, gesture in enumerate(gestures):
        for imageFile in files[i]:

            print('Processing image {}'.format(imageFile))

            # Load image as a numpy array
            img = imread(imageFile)

            # Generate image features
            features[counter] = getGray(img, binary=binary)
            labels[counter, 0] = gesture
            labels[counter, 1] = imageFile

            counter += 1

    dt = round(time.time() - t0, 2)
    print('Completed processing {} images in {}s'.format(counter, dt))
    t0 = time.time()

    # Generate pandas dataframe with labels and features
    dfLabels = pd.DataFrame(labels, columns=['label', 'path'])
    dfFeatures = pd.DataFrame(features, columns=['f' + str(i) for i in range(240*280)])
    df = dfLabels.join(dfFeatures)
    
    # Store dataframe as csv file
    print('Saving data to {}'.format(filename))
    df.to_csv(filename)
##    f = open(filename, 'wb')
##    f.flush()
##    pickle.dump(df, f)
##    f.close() 
    
    dt = round(time.time() - t0, 2)
    print('File {} saved in {}s'.format(filename, dt))
    t0 = time.time()

def generateHOGFeatures(filename='data.csv', binary=False):
    """Reads image files, generates HOG features and saves the features and
    labels in a csv file."""

    t0 = time.time()
    
    gestures = [rps.ROCK, rps.PAPER, rps.SCISSORS]

    # Create a list of image files for each gesture
    files = []
    for i, gesture in enumerate(gestures):
        path = os.path.join(rps.imgPathsRaw[gesture], '*.png')
        files.append(glob(path))
        files[i].sort(key=str.lower)

    nbImages = sum([len(i) for i in files])

    # Create empty numpy arays for features and labels
    features = np.empty((nbImages, 8400), dtype=np.float)
    labels = np.empty((nbImages, 2), dtype=np.object)

    # Generate HOG features and store in arrays
    counter = 0
    for i, gesture in enumerate(gestures):
        for imageFile in files[i]:

            print('Processing image {}'.format(imageFile))

            # Load image as a numpy array
            img = imread(imageFile)

            # Generate HOG features
            features[counter] = getHOG(img, binary=binary)
            labels[counter, 0] = gesture
            labels[counter, 1] = imageFile

            counter += 1

    dt = round(time.time() - t0, 2)
    print('Completed processing {} images in {}s'.format(counter, dt))

    # Generate pandas dataframe with labels and features
    dfLabels = pd.DataFrame(labels, columns=['label', 'path'])
    dfFeatures = pd.DataFrame(features, columns=['f' + str(i) for i in range(8400)])
    df = dfLabels.join(dfFeatures)
    
    # Store dataframe as csv
    print('Saving data to {}'.format(filename))
    df.to_csv(filename)


def getGray(img, hueValue=.36, binary=False):
    """Returns the grayscale of the source image with its background
    removed."""
    
    # Remove background
    if binary:
        img = removeBackgroundBinary(img, hueValue)
    else:
        img = removeBackground(img, hueValue)
        img = color.rgb2gray(img)
    
    return img.ravel()


def getHOG(img, hueValue=.36, binary=False, visualise=False):
    """Returns the HOG features of the source image with its background
    removed."""
    
    # Remove background
    if binary:
        img = removeBackgroundBinary(img, hueValue)
    else:
        img = removeBackground(img, hueValue)
        img = color.rgb2gray(img)
        
    # Extract HOG features
    hogFeatures = feature.hog(img, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), transform_sqrt=True,
                                   block_norm='L2-Hys', visualise=visualise)
    
    return hogFeatures


def hueDistance(img, hueValue):
    """Returns an image where the pixel values correspond to the distance from
       the hue value of the source image pixels and the hueValue argument."""
    
    # Convert image to HSV colorspace
    hsv = color.rgb2hsv(img)
    
    # Calculate hue distance
    dist = np.abs(hsv[:,:,0] - hueValue)
    
    return dist    


def removeBackground(img, hueValue):
    """Returns an image with the background removed based on the hueValue
    argument."""
    
    # Get an image corresponding to the hue distance from the background hue
    # value
    dist = hueDistance(img, hueValue)
    
    # Create a copy of the source image to use as masked image
    masked = img.copy()
    
    # Select background pixels using thresholding and set value to zero (black)
    masked[dist < filters.threshold_mean(dist)] = 0
    
    return masked

def removeBackgroundBinary(img, hueValue):
    """Returns an binary image with black background and white foreground."""
    
    # Get an image corresponding to the hue distance from the background hue
    # value
    dist = hueDistance(img, hueValue)
    
    # Return a binary image using thresholding to separate background and
    # foreground
    return dist > filters.threshold_mean(dist)
