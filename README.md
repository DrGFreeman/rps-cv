# rps-cv
A Rock-Paper-Scissors game using computer vision and machine learning on Raspberry Pi.

By Julien de la Bruère-Terreault (drgfreeman@tuta.io)

![Screenshot](img/doc/screen-2-3.png)

## Summary

### Project origin

This project results from a challenge my son gave me when I was teaching him the basics of computer programming making a simple text based Rock-Paper-Scissors game in Python. At that time I was starting to experiment with computer vision with a Raspberry Pi and an old USB webcam so my son naively asked me:

*"Could you make a Rock-Paper-Scissors game that uses the camera to detect hand gestures?"*

I accepted the challenge and about a year and a lot of learning later, I completed the challenge with a functional game.

### Overview of the game

The game uses a Raspberry Pi computer and Raspberry Pi camera installed on a 3D printed support with LED strips to achieve consistent images.

The pictures taken by the camera are processed and fed to an image classifier that determines whether the gesture corresponds to "Rock", "Paper" or "Scissors" gestures.

The image classifier uses a [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine), a class of [machine learning](https://en.wikipedia.org/wiki/Machine_learning) algorithm. The image classifier has been priorly "trained" with a bank of labeled images corresponding to the "Rock", "Paper", "Scissors" gestures captured with the Raspberry Pi camera.

### How it works

The image below shows the processing pipeline for the training of the image classifier (top portion) and the prediction of gesture for new images captured by the camera during the game (bottom portion). Click [here](https://raw.githubusercontent.com/DrGFreeman/rps-cv/master/img/doc/rps-pipeline.png) for full size image.
![Rock-Paper-Scissors computer vision & machine learning pipeline](img/doc/rps-pipeline.png)

## Dependencies

The project depends on and has been tested with the following libraries:

* OpenCV 3.3.0 with bindings for Python 3*
* Python 3.4+
* Numpy 1.13.0
* Scikit-Learn 0.18.2
* Scikit-Image 0.13.0
* Pygame 1.9.3
* Picamera
* [DrGFreeman/PyTools](https://github.com/DrGFreeman/PyTools) (timer and filter modules)

\* Follow [this guide](https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/) for installation of OpenCV on the Raspberry Pi. Install Python libraries within the same virtual environment as OpenCV using the `pip install <package_name>` command. Picamera is installed by default on [Raspbian](https://www.raspberrypi.org/downloads/raspbian/) images.

### Hardware:

* [Raspberry Pi 3 Model B computer](https://www.raspberrypi.org/products/raspberry-pi-3-model-b/)
* [Raspberry Pi Camera Module V2](https://www.raspberrypi.org/products/camera-module-v2/)
* A green background to allow background subtraction in the captured images.
* A physical setup for the camera to ensure consistent lighting and camera position. The 3D models I used are available [on Thingiverse](https://www.thingiverse.com/thing:2598378).


![Camera & lighting setup](img/doc/hardware1.png)

## Project files

* `capture.py`  
This file opens the camera in "capture mode", to capture and label images that will later be used to train the image classifier. The captured images are automatically named and stored in a folder structure.

* `train.py`*  
This script reads and processes the training images in preparation for training the image classifier. The processed image data is then used to train the support vector machine image classifier. The trained classifier is stored in the `clf.pkl` file read by `play.py`.

* `playgui.py`  
This file runs the actual Rock-Paper-Scissors game using the camera and the trained image classifier in a graphical user interface (GUI). Images from each play are captured and added to the image bank, creating additional images to train the classifier.

* `rpsgui.py`  
This file defines the RPSGUI class and associated methods to manage the game
 graphical user interface (GUI).

* `rpsimgproc.py`  
This file provides the image processing functions used by the various other Python files.

* `rpsutil.py`  
This file provides functions and constants used by the various other Python files.

* `camera.py`  
This file defines the Camera class, a wrapper around the picamera library, with specific methods for the project such as white balance calibration.

* `play.py`  
This file runs the actual Rock-Paper-Scissors game similarly to playgui.py except the game output is done in the terminal and OpenCV window (no GUI).

\* Note that the due to memory limitations on the Raspberry Pi, the `train.py` script may not run properly on the Raspberry Pi with training sets of more than a few hundred images. Consequently, it is recommended to run these on a more powerful computer. This computer must also have OpenCV, Python 3.4+ and the numpy, scikit-learn and scikit-image Python libraries installed.

## Screenshots

### Play mode with Graphical User Interface (`playgui.py`)

Initial screen:  
![Initial screen](img/doc/screen-0-0.png)

Computer wins the play:  
![Computer wins play](img/doc/screen-1-0.png)

Player wins the play:  
![Player wins play](img/doc/screen-2-3.png)

Tie:  
![Tie](img/doc/screen-4-4-tie.png)

Game over, player wins the game:  
![Game over - player wins](img/doc/screen-3-5-game-over.png)

### Image capture mode (`capture.py`)
![Capture mode](img/doc/screen-capture.py.png)

### Play mode without GUI (`play.py`)
![Play no GUI](img/doc/screen-play.py.png)
