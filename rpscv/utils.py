# rpsutil.py
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

# This file defines variables and functions to ensure consistancy in capture and
# naming of images.

import glob
import time

import numpy as np

# Define possible gestures as constants
ROCK = 0
PAPER = 1
SCISSORS = 2

# Define text labels corresponding to gestures
gestureTxt = {ROCK: 'rock', PAPER: 'paper', SCISSORS: 'scissors'}

# Define paths to raw image folders
imgPathsRaw = {ROCK: './img/rock/', PAPER: './img/paper/',
               SCISSORS: './img/scissors/'}

def cameraSetup():
    from rpscv.camera import Camera
    """Returns a camera object with pre-defined settings."""

    # Settings
    size = 8
    frameRate = 40
    awbFilename = 'awb_gains.txt'

    # Create Camera object
    print("Initializing camera")
    cam = Camera(size=size, frameRate=frameRate)

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

class Filter1D:
    """A one dimensional filter class. Useful for real-time filtering of noisy
    time series data such as sensor signal, etc."""

    def __init__(self, maxSize=3):
        """Class constructor. maxSize argument defines the size (number of data
        points) of the signal to be kept. maxSize must be an odd integer >= 3.
        """
        if maxSize % 2 == 1 and maxSize >= 3:
            self._maxSize = maxSize
        else:
            raise ValueError("maxSize must be an odd integer >= 3")
        self._data = np.ndarray(0)

    def addDataPoint(self, dataPoint):
        """Adds new data point(s) to the data array. If the data array size
        exceeds the maxSize attribute, the older data points will be trimmed
        from the array (left trim). dataPoint can be a single point, a list or
        a numpy one dimensional array."""
        ##  Append new data point(s) to end of array
        self._data = np.insert(self._data, self._data.size, dataPoint)
        ##  Trim begining of array if longer than maxSize
        if self._data.size > self._maxSize:
            self._data = self._data[self._data.size - self._maxSize:]

    def getData(self):
        """Returns the complete data array."""
        return self._data

    def getLast(self):
        """Returns the last (most recent) data point from the data array."""
        return self._data[-1]

    def getMean(self, windowSize=0):
        """Returns the mean of the last n points from the data array where n
        equals windowSize. If windowSize is not specified, is set to 0 or is
        greater than maxSize, windowSize will be automatically set to maxSize
        and the mean of the entire data array will be returned."""
        if self._data.size == 0:
            raise RuntimeError("Filter1D data is empty. Call Filter1D.addDataPoint() to add data prior calling Filter1D.getMean().")
        if type(windowSize) is int:
            if windowSize <= 0 or windowSize > self._maxSize:
                windowSize = self._maxSize
            return np.mean(self._data[-windowSize:])
        else:
            raise TypeError("windowSize must be an integer")

    def getMedian(self, windowSize=0):
        """Returns the median of the last n points from the data array where n
        equals windowSize. windowSize must be an odd integer. If windowSize
        is not specified or is set to 0, windowSize will be automatically set
        to maxSize and the median of the entire data array will be returned."""
        if self._data.size == 0:
            raise RuntimeError("Filter1D data is empty. Call Filter1D.addDataPoint() to add data prior calling Filter1D.getMedian().")
        if type(windowSize) is not int:
            raise TypeError("windowSize must be an integer")
        if windowSize <= 0 or windowSize > self._maxSize:
            windowSize = self._maxSize
        if windowSize % 2 == 1 and windowSize <= self._maxSize:
            return np.median(self._data[-windowSize:])
        else:
            raise ValueError("windowSize must be an odd integer <= maxSize")

class Timer:

    def __init__(self):
        """A timer that can be used to measure elapsed time, manage time steps
        in loops, control execution times, etc.
        The constructor, starts the timer at instantiation."""
        self.paused = False
        self.pauseInitTime = None
        self.pauseElapsed = 0
        self.initTime = time.time()

    def getElapsed(self):
        """Returns the time elapsed since instantiation or last reset minus sum
        of paused time."""
        if self.paused:
            return self.pauseInitTime - self.initTime - self.pauseElapsed
        else:
            return time.time() - self.initTime - self.pauseElapsed

    def isWithin(self, delay):
        """Returns True if elapsed time is within (less than) delay argument.
        This method is useful to control execution of while loops for a fixed
        time duration."""
        if self.getElapsed() < delay:
            return True
        else:
            return False

    def pause(self):
        """Pauses the timer."""
        self.pauseInitTime = time.time()
        self.paused = True

    def reset(self):
        """Resets the timer initial time to current time."""
        self.paused = False
        self.pauseInitTime = None
        self.pauseElapsed = 0
        self.initTime = time.time()

    def resume(self):
        """Resumes the timer following call to .pause() method."""
        if self.paused:
            self.pauseElapsed += time.time() - self.pauseInitTime
            self.paused = False
        else:
            print("Warning: Timer.resume() called without prior call to Timer.pause()")

    def sleepToElapsed(self, delay, reset = True):
        """Sleeps until elapsed time reaches delay argument. If reset argument
        is set to True (default), the timer will also be reset. This method is
        useful to control fixed time steps in loops."""
        if self.getElapsed() < delay:
            time.sleep(delay - self.getElapsed())
        if reset:
            self.reset()
