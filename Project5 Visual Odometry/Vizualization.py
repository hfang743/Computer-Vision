import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

import imutils
import os
import sys
import random
import pickle

from imageCorrection import *


def playViz(var, var2, frameset, startframe, NFrames):
    l = openVar(var)
    cl = openVar(var2)
    coordinates = np.array(l)

    # fig2 = plt.figure()
    coordx = coordinates[:, 0]
    coordy = coordinates[:, 1]
    maxX = np.max(np.abs(coordx)) + 10
    maxY = np.max(np.abs(coordy)) + 10

    Ccoordinates = np.array(cl)[startframe - 19:NFrames - 19]

    Ccoordx = Ccoordinates[:, 0]
    Ccoordy = Ccoordinates[:, 1]
    CmaxX = np.max(np.abs(Ccoordx)) + 100
    CmaxY = np.max(np.abs(Ccoordy)) + 100

    plt.ion()
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True, sharex=True)
    Cx, Cy = [], []
    Cc = ax2.scatter(Cx, Cy, color='y')
    ax2.set_title('OpenCV functions')
    plt.xlim(-CmaxX, CmaxX)
    plt.ylim(-CmaxY, CmaxY)

    # fig1, ax1 = plt.subplots()

    x, y = [], []
    ax1.set_title('Custom Functions')
    sc = ax1.scatter(x, y, color='r')
    # plt.xlim(-maxX, maxX)
    # plt.ylim(-maxY, maxY)

    plt.draw()
    i = 0
    for j in range(startframe, NFrames):

        Cx.append(Ccoordx[i])
        Cy.append(Ccoordy[i])
        Cc.set_offsets(np.c_[Cx, Cy])

        x.append(coordx[i])
        y.append(coordy[i])
        sc.set_offsets(np.c_[x, y])
        i += 1

        fig1.canvas.draw_idle()
        plt.pause(0.0001)
        workingframe = imutils.resize(frameset[j], width=320 * 2, height=180 * 2)
        cv2.imshow('Output', workingframe)
        #
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def showTracking(var, cv, startframe, NFrames):
    l = openVar(var)
    l = np.array(l)
    lLen = len(l)
    # print(lLen)
    plt.scatter(l[:, 0], l[:, 1], color='r')
    plt.scatter(l[0, 0], l[0, 1], color='g')
    plt.scatter(l[len(l) - 1, 0], l[len(l) - 1, 1], color='b')

    cl = openVar(cv)
    cl = np.array(cl)[startframe - 19:NFrames - 19]
    plt.scatter(cl[:, 0], cl[:, 1], color='y')
    plt.scatter(cl[0, 0], cl[0, 1], color='k')
    plt.scatter(cl[len(cl) - 1, 0], cl[len(cl) - 1, 1], color='teal')
