# play.py
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

# This file is the main program to run to play the Rock-Paper-Scissors game.
# Game output is made through the terminal and OpenCV window (no GUI).

import time
import random

import cv2
import numpy as np

from rpscv import utils
from rpscv import imgproc as imp

import pickle

def saveImage(img, gesture, notify=False):

    # Define image path and filename
    folder = utils.imgPathsRaw[gesture]
    name = utils.gestureTxt[gesture] + '-' + time.strftime('%Y%m%d-%H%M%S')
    extension = '.png'

    if notify:
        print('Saving {}'.format(folder + name + extension))

    # Save image
    cv2.imwrite(folder + name + extension, img)

try:
    # Load classifier from pickle file
    filename = 'clf.pkl'
    with open(filename, 'rb') as f:
        clf = pickle.load(f)

    # Create camera object with pre-defined settings
    cam = utils.cameraSetup()

    # Initialize variable to stop while loop execution
    stop = False

    # Initialize opencv GUI window (resizeable)
    cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)

    # Print instructions
    print("\nImage recognition mode")
    print("Press ESC or q to quit\n")

    # Initialize last gesture value
    lastGesture = -1

    # Initialize player scores
    playerScore = 0
    computerScore = 0
    endScore = 5

    # Main loop
    while not stop:
        # Capture image from camera
        img = cam.getOpenCVImage()

        # Crop image
        img = imp.crop(img)

        # Convert image to RGB (from BGR)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get grayscale image
        gray = imp.getGray(imgRGB, threshold=17)

        # Count non-background pixels
        nonZero = np.count_nonzero(gray)

        # Define waiting time for cv2.waitKey()
        waitTime = 1

        # Parameters for saving new images
        gesture = None
        notify = False

        #if  9000 < nz and nz < 25000:
        if nonZero > 9000:

            # Predict gesture
            predGesture = clf.predict([gray])[0]

            if predGesture == lastGesture:
                successive += 1
            else:
                successive = 0

            if successive == 2:
                print('Player: {}'.format(utils.gestureTxt[predGesture]))
                waitTime=3000
                gesture = predGesture

                # Computer gesture
                computerGesture = random.randint(0,2)
                print('Computer: {}'.format(utils.gestureTxt[computerGesture]))

                diff = computerGesture - predGesture
                if diff in [-2, 1]:
                    print('Computer wins!')
                    computerScore += 1
                elif diff in [-1, 2]:
                    print('Player wins!')
                    playerScore += 1
                else:
                    print('Tie')
                print('Score: player {}, computer {}\n'.format(playerScore,
                                                             computerScore))

            lastGesture = predGesture

        else:

            lastGesture = -1

        # Rotate and add framerate to copy of image
        imgFR = imp.fastRotate(img)
        txtPos = (5, imgFR.shape[0] - 10)
        cam.addFrameRateText(imgFR, txtPos, bgr=(0,0,255))

        # Display image
        cv2.imshow('Camera', imgFR)

        # Wait for key press
        key = cv2.waitKey(waitTime)
        if key in [27, 113]:
            # Escape or "Q" key pressed; Stop.
            stop = True
        elif key == 114:
            # R key pressed (Rock)
            gesture = utils.ROCK
            notify = True
        elif key == 112:
            # P key pressed (Paper)
            gesture = utils.PAPER
            notify = True
        elif key in [115, 99]:
            # S or C key pressed (Scissors)
            gesture = utils.SCISSORS
            notify = True

        if gesture is not None:
            # Save new image
            saveImage(img, gesture, notify)

        if playerScore == endScore or computerScore == endScore:
            stop = True
            if computerScore > playerScore:
                print('Game over, computer wins...')
            else:
                print('Game over, player wins!!!')

finally:
    f.close()
    cv2.destroyAllWindows()
    cam.close()
