# rpsutil.py
# Source: https://github.com/DrGFreeman/rps-cv
#
# MIT License
#
# Copyright (c) 2017 Julien de la Bruere-Terreault <drgfreeman@tuta.io>
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

# This file defines variables and functions to ensure consistancy in capture and
# naming of images.

import glob
import time

import numpy as np

#import camera

# Define possible gestures as constants
ROCK = 1
PAPER = 2
SCISSORS = 3

# Define text labels corresponding to gestures
gestureTxt = {ROCK: 'rock', PAPER: 'paper', SCISSORS: 'scissors'}

# Define paths to raw image folders
imgPathsRaw = {ROCK: './img/rock/', PAPER: './img/paper/',
            SCISSORS: './img/scissors/'}

def cameraSetup():
    import camera
    """Returns a camera object with pre-defined settings."""

    # Settings
    size = 8
    frameRate = 40
    awbFilename = 'awb_gains.txt'

    # Create Camera object
    print("Initializing camera")
    cam = camera.Camera(size=size, frameRate=frameRate)

    # Check if white balance file exists
    if len(glob.glob(awbFilename)) != 0:
        # File exists, set camera white balance using gains from file
        print("Reading white balance gains from {}".format(awbFilename))
        cam.readWhiteBalance(awbFilename)
    else:
        # File does not exist. Prompt user to perform white balance.
        print("WARNING: No white balance file found. ")
        if input("Perform white balance (Y/n)?\n") != "n":
            print("Performing white balance.")
            print("Place a sheet of white paper in front of camera.")
            input("Press any key when ready.\n")
            cam.doWhiteBalance(awbFilename)

    return cam

def crop(img):
    """Returns a cropped image to pre-defined shape."""
    return img[75:275, 125:425]

def fastRotate(img):
    """Rotates the image clockwise 90 deg."""
    return np.transpose(img.copy(), axes=(1, 0, 2))[:,::-1,:]
