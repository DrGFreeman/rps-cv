# rpsimgproc.py
# Source: https://github.com/DrGFreeman/rps-cv
#
# MIT License
#
# Copyright (c) 2017-2019 Julien de la Bruere-Terreault <drgfreeman@tuta.io>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file defines functions for processing of images.

import os
from glob import glob
import time

import numpy as np

from skimage.io import imread
from skimage import color
from skimage import feature
from skimage import filters

from rpscv import utils

import cv2

def crop(img):
    """Returns a cropped image to pre-defined shape."""
    return img[75:275, 125:425]

def fastRotate(img):
    """Rotates the image clockwise 90 deg."""
    return np.transpose(img, axes=(1, 0, 2))[:,::-1,:].copy()

def generateGrayFeatures(imshape=(200,300, 3), nbImg=0, verbose=False, rs=42):
    """Reads training image files, generates features from grayscale image and
    saves the features and labels in a csv file to be used to train the image
    classifier."""

    imsize = imshape[0] * imshape[1]

    t0 = time.time()

    gestures = [utils.ROCK, utils.PAPER, utils.SCISSORS]

    # Create a list of image files for each gesture
    files = []
    for i, gesture in enumerate(gestures):
        path = os.path.join(utils.imgPathsRaw[gesture], '*.png')
        files.append(glob(path))
        files[i].sort(key=str.lower)

    nbImages = sum([len(i) for i in files])

    # Create empty numpy arays for features and labels
    features = np.empty((nbImages, imsize), dtype=np.float32)
    labels = np.empty((nbImages), dtype=np.int)

    # Generate grayscale images
    counter = 0
    for i, gesture in enumerate(gestures):

        if nbImg > 0:
            np.random.seed = rs
            files[i] = np.random.permutation(files[i])
            if len(files[i]) > nbImg:
                files[i] = files[i][:int(nbImg / 3)]

        for imageFile in files[i]:

            if verbose:
                print('Processing image {}'.format(imageFile))

            # Load image as a numpy array
            img = imread(imageFile)

            if img.shape == imshape:

                # Generate and store image features in features array
                features[counter] = getGray(img, threshold=17)

                # Store image label in labels array
                labels[counter] = gesture

                counter += 1

            else:
                print('Image {} has invalid shape: {}, {} expected, skipping image.'.format( \
                    imageFile, img.shape, imshape))

    print('Completed processing {} images'.format(counter))

    return features[:counter], labels[:counter]


def getGray(img, hueValue=63, threshold=0):
    """Returns the grayscale of the source image with its background
    removed as a 1D feature vector."""

    img = removeBackground(img, hueValue, threshold)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255

    return img.ravel()


def hueDistance(img, hueValue):
    """Returns an image where the pixel values correspond to the distance from
       the hue value of the source image pixels and the hueValue argument."""

    # Convert image to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    hChannel = hsv[:,:,0].astype(int)

    # Calculate hue distance
    dist = np.abs(hsv[:,:,0] - hueValue)
    if hueValue < 90:
        hueOffset = 180
    else:
        hueOffset = -180

    dist = np.minimum(np.abs(hChannel - hueValue),
                      np.abs(hChannel - (hueValue + hueOffset)))

    return dist


def removeBackground(img, hueValue, threshold=0):
    """Returns an image with the background removed based on the hueValue
    argument."""

    # Get an image corresponding to the hue distance from the background hue
    # value
    dist = hueDistance(img, hueValue)

    # Create a copy of the source image to use as masked image
    masked = img.copy()

    # Select background pixels using thresholding and set value to zero (black)
    if threshold == 0:
        masked[dist < filters.threshold_mean(dist)] = 0
    else:
        masked[dist < threshold] = 0

    return masked
