# capture.py
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

# This script is opens the camera to capture images corresponding to Rock, Paper
# Scisors gestures in a consistant format. It is to be used to capture the images
# used to train the classifier.

import time

import cv2
import numpy as np

import rpsutil as rps

def saveImage(img, gesture):

    # Define image path and filename
    folder = rps.imgPathsRaw[gesture]
    name = rps.gestureTxt[gesture] + '-' + time.strftime('%Y%m%d-%H%M%S')
    extension = '.png'

    print("Saving " + name + extension + " - Accept ([y]/n)?")

    # Write gesture name to image and show for a few seconds
    imgTxt = img.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(imgTxt, rps.gestureTxt[gesture], (10,25), font, 1, (0, 0, 255))
    cv2.imshow('Camera', imgTxt)
    key = cv2.waitKey(50)
    if key not in [110, 120]:
        # Key is not x or n. Save image
        cv2.imwrite(folder + name + extension, img)
        print("Saved ({}x{})".format(img.shape[1], img.shape[0]))
    else:
        print("Save cancelled")

try:
    # Create camera object with pre-defined settings
    cam = rps.cameraSetup()

    # Initialize variable to stop while loop execution
    stop = False

    # Initialize opencv GUI window (resizeable)
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    # Print instructions
    print("\nImage capture mode")
    print("Press the following keys to capture:")
    print("Rock gesture: r")
    print("Paper gesture: p")
    print("Scisors gesture: s or c")
    print("Press ESC or q to quit capture mode\n")

    # Main loop
    while not stop:
        # Capture image from camera
        img = cam.getOpenCVImage()

        # Crop image
        img = rps.crop(img)

        # Add framerate to copy of image
        imgFR = img.copy()
        #imgFR = rmBg(imgFR)
        txtPos = (5, img.shape[0] - 10)
        cam.addFrameRateText(imgFR, txtPos, bgr=(0,0,255))

        # Display image
        cv2.imshow('Camera', imgFR)

        # Wait for key press
        key = cv2.waitKey(1)
        if key in [27, 113]:
            # Escape or "Q" key pressed; Stop.
            stop = True
        else:
            gesture = None
            if key == 114:
                # "R" key pressed (Rock)
                gesture = rps.ROCK
            elif key == 112:
                # "P" key pressed (Paper)
                gesture = rps.PAPER
            elif key in [115, 99]:
                # "S" or "C" key pressed (Scisors)
                gesture = rps.SCISSORS
            if gesture:
                saveImage(img, gesture)

finally:
    cv2.destroyAllWindows()
    cam.close()
