import time

import cv2
import numpy as np

import rpsutil as rps
import rpsimgproc as imp

import pickle

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

    # Initialize count of successive similar gestures

    # Main loop
    while not stop:
        # Capture image from camera
        img = cam.getOpenCVImage()

        # Crop image
        img = rps.crop(img)

        # Get grayscale image
        gray = imp.getGray(img)

        # Count non-background pixels
        nz = np.count_nonzero(gray)

        if  8000 < nz and nz < 30000:

            # Predict gesture
            predGesture = clf.predict([gray])[0]
        
            #print(rps.gestureTxt[predGesture])

            if predGesture == lastGesture:
                successive += 1
            else:
                successive = 0

            if successive == 3:
                print('Locked gesture: {}'.format(rps.gestureTxt[predGesture]))
                time.sleep(3)

            lastGesture = predGesture

        else:
            #print('BG')

            lastGesture = 0

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

finally:
    f.close()
    cv2.destroyAllWindows()
    cam.close()
