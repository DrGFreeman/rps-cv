import time

import cv2
import numpy as np

import imutils as imu

import rpsutil as rps

try:
    cam = rps.cameraSetup()

    stop = False

    while not stop:

        img = cam.getOpenCVImage()
        img = imu.rotate_bound(img, 90)
        txtPos = (0, int(img.shape[0] - 10))
        cam.addFrameRateText(img, txtPos)
        cv2.imshow('Camera', img)
        key = cv2.waitKey(1)
        if key in [27, 113]:
            stop = True

except Exception:
    raise

finally:
    cv2.destroyAllWindows()
    cam.close()
