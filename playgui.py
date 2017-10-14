import pickle
import random
#import sys
import time

import pygame as pg
import pygame.locals

import numpy as np
import cv2

import rpsutil as rps
import rpsimgproc as imp
import rpsgui

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

    # Initialize last gesture value
    lastGesture = 0

    # Define score at which game ends
    endScore = 5

    # Initialize GUI
    gui = rpsgui.RPSGUI()

    # Load static images for computer gestures
    coImgs = {}
    img = cv2.imread('img/gui/rock.png', cv2.IMREAD_COLOR)
    coImgs[rps.ROCK] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread('img/gui/paper.png', cv2.IMREAD_COLOR)
    coImgs[rps.PAPER] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread('img/gui/scissors.png',
                     cv2.IMREAD_COLOR)
    coImgs[rps.SCISSORS] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load green image
    greenImg = cv2.imread('img/gui/green.png', cv2.IMREAD_COLOR)
    greenImg = cv2.cvtColor(greenImg, cv2.COLOR_BGR2RGB)

    while True:

        # Get image from camera
        img = imp.crop(cam.getOpenCVImage())

        # Convert image to RGB (from BGR)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Set player image to imgRGB
        gui.setPlImg(imgRGB)

        # Get grayscale image
        gray = imp.getGray(imgRGB, threshold=.1)

        # Count non-background pixels
        nonZero = np.count_nonzero(gray)

        # Define waiting time
        waitTime = 0

        # Parameters for saving new images
        gesture = None
        notify = False

        # Check if player hand is present
        if nonZero > 9000:

            # Predict gesture
            predGesture = clf.predict([gray])[0]

            if predGesture == lastGesture:
                successive += 1
            else:
                successive = 0

            if successive == 2:
                print('Player: {}'.format(rps.gestureTxt[predGesture]))
                waitTime = 3000
                gesture = predGesture

                # Computer gesture
                computerGesture = random.randint(1,3)
                print('Computer: {}'.format(rps.gestureTxt[computerGesture]))

                # Set computer image to computer gesture
                gui.setCoImg(coImgs[computerGesture])

                diff = computerGesture - predGesture
                if diff in [-2, 1]:
                    print('Computer wins!')
                    gui.setWinner('computer')
                elif diff in [-1, 2]:
                    print('Player wins!')
                    gui.setWinner('player')
                else:
                    print('Tie')
                    gui.setWinner('tie')
                print('Score: player {}, computer {}\n'.format(gui.plScore,
                                                             gui.coScore))

            lastGesture = predGesture

        else:

            lastGesture = 0

            # Set computer image to green
            gui.setCoImg(greenImg)
            gui.setWinner()

        # Draw GUI
        gui.draw()

        # Flip pygame display
        pg.display.flip()

        # Wait
        pg.time.wait(waitTime)

        if gesture:
            # Save new image
            saveImage(img, gesture, notify)

        # Check if scores reach endScore (end of game)
        if gui.plScore == endScore or gui.coScore == endScore:
            if gui.coScore > gui.plScore:
                print('Game over, computer wins...')
            else:
                print('Game over, player wins!!!')
            gui.quit()

        # Check pygame events
        for event in pg.event.get():
            if event.type == pg.locals.QUIT:
                gui.quit()

    gui.quit()

finally:
    f.close()
    cam.close()
