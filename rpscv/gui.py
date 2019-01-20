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

import cv2
import pygame as pg
import pygame.freetype

class RPSGUI():

    def __init__(self, privacy=False, loop=False):
        pg.init()
        self.privacy = privacy
        self.loop = loop
        self.sWidth = 640
        self.sHeight = 480
        self.surf = pg.display.set_mode((self.sWidth, self.sHeight))
        pg.display.set_caption('Rock-Paper-Scissors by drgfreeman@tuta.io')
        self.plScore = 0
        self.coScore = 0
        self.plImg = pg.Surface((200, 300))
        self.coImg = pg.Surface((200, 300))
        self.plImgPos = (380, 160)
        self.coImgPos = (60, 160)
        self.plZone = pg.Surface((250, 330))
        self.coZone = pg.Surface((250, 330))
        self.plZonePos = (355, 145)
        self.coZonePos = (35, 145)
        self.winner = None

        # colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        self.showPrivacyNote()

    def blitTextAlignCenter(self, surf, text, pos):
        tWidth = text[1].width
        surf.blit(text[0], (pos[0] - tWidth / 2, pos[1]))

    def draw(self):
        # Fill surface with background color
        self.surf.fill(self.WHITE)

        # Draw boxes around computer and player areas
        plVertices = [(325, 3), (634, 3), (634, 476), (325, 476), (325, 3)]
        pg.draw.polygon(self.surf, self.BLACK, plVertices, 1)
        coVertices = [(5, 3), (315, 3), (315, 476), (5, 476), (5, 3)]
        pg.draw.polygon(self.surf, self.BLACK, coVertices, 1)

        # Render computer and player text
        font = pg.freetype.SysFont(None, 30)
        text = font.render('PLAYER', self.BLACK)
        self.blitTextAlignCenter(self.surf, text, (480,15))
        text = font.render('COMPUTER', self.BLACK)
        self.blitTextAlignCenter(self.surf, text, (160,15))

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
        font = pg.freetype.SysFont(None, 100)
        text = font.render(str(self.plScore), self.BLACK)
        self.blitTextAlignCenter(self.surf, text, (480, 60))
        text = font.render(str(self.coScore), self.BLACK)
        self.blitTextAlignCenter(self.surf, text, (160, 60))

    def gameOver(self, delay=3500):
        # Create surface for Game Over message
        goZone = pg.Surface((400, 200))

        # Fill surface with background color
        goZone.fill(self.WHITE)

        # Draw box around surface
        vertices = [(3, 3), (396, 3), (396, 196), (3, 196), (3, 3)]
        pg.draw.polygon(goZone, self.BLACK, vertices, 1)

        # Render text on surface
        font = pg.freetype.SysFont(None, 40)
        gameOverText = font.render('GAME OVER', self.BLACK)
        self.blitTextAlignCenter(goZone, gameOverText, (200, 45))

        if self.plScore > self.coScore:
            winner = 'PLAYER'
            color = self.GREEN
        else:
            winner = 'COMPUTER'
            color = self.RED

        winnerText = font.render('{} WINS!'.format(winner), color)
        self.blitTextAlignCenter(goZone, winnerText, (200, 110))

        # Blit goZone to main surface
        pos = (self.sWidth / 2 - 200, 175)
        self.surf.blit(goZone, pos)

        pg.display.flip()

        pg.time.wait(delay)

        if self.loop:
            self.reset()
        else:
            self.quit()

    def showPrivacyNote(self, delay=10000):
        if self.privacy:
            # Fill surface with background color
            self.surf.fill(self.WHITE)

            #Render text on surface
            font = pg.freetype.SysFont(None, 40)
            text = font.render('Privacy Notice', self.RED)
            pos = (self.sWidth / 2, 100)
            self.blitTextAlignCenter(self.surf, text, pos)

            font = pg.freetype.SysFont(None, 20)
            pn = ['Images captured during the game are stored to help']
            pn.append('improve the image classification algorithm and may be')
            pn.append('shared publicly. By playing this game you agree to have')
            pn.append('images of your hand captured and stored.')
            for i, line in enumerate(pn):
                text = font.render(line, self.BLACK)
                pos = (self.sWidth / 2, 150 + 25 * i)
                self.blitTextAlignCenter(self.surf, text, pos)

            pg.display.flip()
            pg.time.wait(delay)

    def quit(self, delay=0):
        pg.time.wait(delay)
        pg.quit()
        sys.exit()

    def reset(self):
        self.plScore = 0
        self.coScore = 0
        self.showPrivacyNote()

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
