# camera.py
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

# This module defines the Camera class, a wrapper for the Raspberry Pi camera
# based on the picamera library to be used with OpenCV computer vision library.

import time

import cv2
import numpy as np
#from picamera import PiCamera, PiCameraCircularIO

from rpscv.utils import Filter1D, Timer

class Camera():

    def __init__(self, size=10, frameRate=40, hflip=False, vflip=False):
        """A wrapper class for the Raspberry Pi camera using the picamera
        python library. The size parameter sets the camera resolution to
        size * (64, 48)."""
        from picamera import PiCamera, PiCameraCircularIO
        self.active = False
        try:
            if type(size) is not int:
                raise TypeError("Size must be an integer")
            elif 1 <= size and size <= 51:
                self.size = size
                self.hRes = size * 64
                self.vRes = size * 48
            else:
                raise ValueError("Size must be in range 1 to 51")
        except TypeError or ValueError:
            raise
        self.picam = PiCamera()
        self.picam.resolution = (self.hRes, self.vRes)
        self.picam.framerate = frameRate
        self.picam.hflip = hflip
        self.picam.vflip = vflip
        time.sleep(1)
        self.stream = PiCameraCircularIO(self.picam, seconds=1)
        self.frameRateTimer = Timer()
        self.frameRateFilter = Filter1D(maxSize=21)
        self.start()

    def close(self):
        """Stops the running thread and closes the PiCamera instance."""
        self.stop()
        self.picam.close()

    def doWhiteBalance(self, awbFilename='awb_gains.txt', mode='auto'):
        """A method that performs white balance calibration, sets the PiCamera
        awb_gains to fixed values and write these values in a file. For best
        results, put a white objet in the camera field of view (a sheet of paper
        ) during the calibration process."""
        ##  Set AWB mode for calibration
        self.picam.awb_mode = mode
        print('Calibrating white balance gains...')
        time.sleep(1)
        ##  Read AWB gains
        gRed = 0
        gBlue = 0
        nbReadings = 100
        for i in range(nbReadings):
            gains = self.picam.awb_gains
            gRed += gains[0]
            gBlue += gains[1]
            time.sleep(.01)
        gains = gRed / nbReadings, gBlue / nbReadings
        ##  Set AWB mode to off (manual)
        self.picam.awb_mode = 'off'
        ##  Set AWB gains to remain constant
        self.picam.awb_gains = gains

        ##  Write AWB gains to file
        gRed = float(gains[0])
        gBlue = float(gains[1])
        f = open(awbFilename, 'w')
        f.flush()
        f.write(str(gRed) + ', ' + str(gBlue))
        f.close()
        print('AWB gains set to:', gRed, gBlue)
        print('AWB gains written to ' + awbFilename)

    def addFrameRateText(self, img, pos=(0, 25), bgr=(0,255,0), samples=21):
        """Returns an image with the frame rate added as text on the image
        passed as argument. The framerate is calculated based on the time
        between calls to this method and averaged over a number of samples.
        img: image to which the framerate is to be added,
        bgr: tuple defining the blue, green and red values of the text color,
        samples: number of samples used for averaging.
        """
        # Calculate framerate and reset timer
        self.frameRateFilter.addDataPoint(1 / self.frameRateTimer.getElapsed())
        self.frameRateTimer.reset()
        # Get averaged framerate as a string
        frString = '{}fps'.format(str(int(round(self.frameRateFilter.getMean(),
                                                0))))
        # Add text to image
        cv2.putText(img, frString, pos, cv2.FONT_HERSHEY_DUPLEX, 1, bgr)

    def getOpenCVImage(self):
        """Grabs a frame from the camera and returns an OpenCV image array."""
        img = np.empty((self.vRes * self.hRes * 3), dtype=np.uint8)
        self.picam.capture(img, 'bgr', use_video_port=True)
        return img.reshape((self.vRes, self.hRes, 3))

    def readWhiteBalance(self, awbFilename='awb_gains.txt'):
        """Reads white balance gains from a file created using the
        .doWitheBalance() method and fixes the PiCamera awb_gains parameter
        to these values."""
        ##  Read AWB gains from file
        f = open(awbFilename, 'r')
        line = f.readline()
        f.close()
        gRed, gBlue = [float(g) for g in line.split(', ')]
        ##  Set AWB mode to off (manual)
        self.picam.awb_mode = 'off'
        ##  Set AWB gains to remain constant
        self.picam.awb_gains = gRed, gBlue
        print('AWB gains set to:', gRed, gBlue)

    def start(self):
        """Starts continuous recording of the camera into a PicameraCircularIO
        buffer."""
        if not self.active:
            self.active = True
            self.picam.start_recording(self.stream, format='h264',
                                       resize=(self.hRes, self.vRes))

    def startPreview(self):
        """Starts the preview of the PiCamera. Works only on the display
        connected directly on the Raspberry Pi."""
        self.picam.start_preview()

    def stop(self):
        """Stops the camera continuous recording and stops the preview if
        active."""
        self.active = False
        self.picam.stop_recording()
        self.stopPreview()

    def stopPreview(self):
        """Stops the PiCamera preview if active."""
        self.picam.stop_preview()
