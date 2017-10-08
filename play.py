# play.py
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

# This file is the main program to run to play the Rock-Paper-Scissors game.

import time
import random

import cv2
import numpy as np

import rpsutil as rps
import rpsimgproc as imp

import pickle

def saveImage(img, gesture, notify=False):

    # Define image path and filename
    folder = rps.imgPathsRaw[gesture]
    name = rps.gestureTxt[gesture] + '-' + time.strftime('%Y%m%d-%H%M%S')
    extension = '.png'

    if notify:
        print('Saving {}'.format(folder + name + extension))

    # Save image
    cv2.imwrite(folder + name + extension, img)

try:
    # Load classifier from pickle file
    filename = 'clf.pkl'
    f = open(filename, 'rb')
    clf = pickle.load(f)
    f.close()
    
    # Create camera object with pre-defined settings
    cam = rps.cameraSetup()

    # Initialize variable to stop while loop execution
    stop = False

    # Initialize opencv GUI window (resizeable)
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    # Print instructions
    print("\nImage recognition mode")
    print("Press ESC or q to quit\n")

    # Initialize last gesture value
    lastGesture = 0

    # Initialize player scores
    playerScore = 0
    computerScore = 0
    endScore = 5

    # Main loop
    while not stop:
        # Capture image from camera
        img = cam.getOpenCVImage()

        # Crop image
        img = rps.crop(img)

        # Get grayscale image
        gray = imp.getGray(img, threshold=.06)

        # Count non-background pixels
        nz = np.count_nonzero(gray)

        # Define waiting time for cv2.waitKey()
        waitTime = 1

        # Parameters for saving new images
        gesture = None
        notify = False

        if  8000 < nz and nz < 30000:

            # Predict gesture
            predGesture = clf.predict([gray])[0]
        
            #print(rps.gestureTxt[predGesture])

            if predGesture == lastGesture:
                successive += 1
            else:
                successive = 0

            if successive == 3:
                print('Player: {}'.format(rps.gestureTxt[predGesture]))
                waitTime=3000
                gesture = predGesture

                # Computer gesture
                computerGesture = random.randint(1,3)
                print('Computer: {}'.format(rps.gestureTxt[computerGesture]))

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

            lastGesture = 0

        # Add framerate to copy of image
        imgFR = img.copy()
        #imgFR = rmBg(imgFR)
        txtPos = (5, img.shape[0] - 10)
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
            gesture = rps.ROCK
            notify = True
        elif key == 112:
            # P key pressed (Paper)
            gesture = rps.PAPER
            notify = True
        elif key in [115, 99]:
            # S or C key pressed (Scissors)
            gesture = rps.SCISSORS
            notify = True

        if gesture:
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
