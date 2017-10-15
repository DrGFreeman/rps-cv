# rpsgui.py
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

# This file defines the RPSGUI class and associated methods to manage the game
# graphical user intrerface (GUI).

import sys
import numpy as np

import pygame as pg
import pygame.freetype

import cv2

class RPSGUI():

    def __init__(self):
        pg.init()
        self.sWidth = 640
        self.sHeight = 480
        self.surf = pg.display.set_mode((self.sWidth, self.sHeight))
        pg.display.set_caption('Rock-Paper-Scissors by drgfreeman@tuta.io')
        self.plScore = 0
        self.coScore = 0
        self.plImg = pg.Surface((200, 300))
        self.coImg = pg.Surface((200, 300))
        self.plImgPos = (380, 165)
        self.coImgPos = (60, 165)
        self.plZone = pg.Surface((250, 330))
        self.coZone = pg.Surface((250, 330))
        self.plZonePos = (355, 150)
        self.coZonePos = (35, 150)
        self.winner = None
        self.plcoFont = pg.freetype.SysFont('', 30)
        self.scoreFont = pg.freetype.SysFont('', 100)

        # colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

    def draw(self):
        # Fill surface with background color
        self.surf.fill(self.WHITE)

        # Render computer and player text
        self.plcoFont.render_to(self.surf, (415, 10), 'PLAYER',
                                 fgcolor=self.BLACK)
        self.plcoFont.render_to(self.surf, (75, 10), 'COMPUTER',
                                 fgcolor=self.BLACK)

        # Set computer and player zone colors
        if self.winner == 'player':
            self.plZone.fill(self.GREEN)
            self.coZone.fill(self.RED)
        elif self.winner == 'computer':
            self.plZone.fill(self.RED)
            self.coZone.fill(self.GREEN)
        elif self.winner == 'tie':
            self.plZone.fill(self.BLUE)
            self.coZone.fill(self.BLUE)
        else:
            self.plZone.fill(self.WHITE)
            self.coZone.fill(self.WHITE)

        # Blit computer and player zone
        self.surf.blit(self.plZone, self.plZonePos)
        self.surf.blit(self.coZone, self.coZonePos)

        # Blit computer and player images
        self.surf.blit(self.plImg, self.plImgPos)
        self.surf.blit(self.coImg, self.coImgPos)

        # Render computer and player scores
        self.scoreFont.render_to(self.surf, (452,65), str(self.plScore),
                                 fgcolor=self.BLACK)
        self.scoreFont.render_to(self.surf, (132,65), str(self.coScore),
                                 fgcolor=self.BLACK)

    def quit(self, delay=0):
        pg.time.wait(delay)
        pg.quit()
        sys.exit()

    def setCoImg(self, img):
        self.coImg = pg.surfarray.make_surface(img[:,::-1,:])

    def setPlImg(self, img):
        self.plImg = pg.surfarray.make_surface(img[::-1,:,:])

    def setWinner(self, winner=None):
        self.winner = winner
        if winner == 'player':
            self.plScore += 1
        elif winner == 'computer':
            self.coScore += 1
