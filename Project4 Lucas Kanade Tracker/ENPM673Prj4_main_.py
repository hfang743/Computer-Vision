import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils
import os
import random
from scipy import signal

import featureRegistration as fr
import imagesProcessing as ip

from imageFileNames import *

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True
flag_trackingFeature = True


def runVidOp(directory, start, rect):
    # tempt, rect, rectT = createtemplate(directory, rect)
    frames = ip.firstTemplateAndSecondFrameAndImageList(
        directory, flag_covertToGray=False)  # get first bounding box around template, template, frames
    frame_contains_template = frames[start]
    left = rect[0]
    right = rect[1]
    top = rect[2]
    bottom = rect[3]

    rect = np.array([[left, right, right], [top, top, bottom], [1, 1, 1]])
    template = ip.subImageInBoundingBoxAndEq(frame_contains_template, rect)
    print("Import images done")
    print('Please Close the initial windows to begin the tracker')
    ip.drawRect(frames[start], rect, flag_show=True, flag_hold=True)
    return frames[start:], template, rect


def main(prgRun):
    # start file
    problem = 3
    problem = int(input('Which problem would you like to run?\nEnter 1 for Bolt\nEnter 2 for Car\nEnter 3 for Dragon Baby\nPlease enter your choice :'))

    if problem == 1:
        directory = './Bolt2/img'
        directory = str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows". Example: ./Bolt2/img: \n'))
        # left = 250, right = 320, top = 75, bottom = 150
        rect = [240, 265, 70, 120]    # left, right, up, bottom bound
        frames, template, rect = runVidOp(directory, start=20, rect=rect)
        fr.LKRegisteration(frames, template, rect, rotate=1, robust=False, numberOfiteration=1000, delta_p_threshold=0.1)  # rect update
        print("Problem 1 finished")

    if problem == 2:
        directory = './Car4/img'
        directory = str(input('What is the name and directory of the Car folder with the images? Note, this should be entered as"./folder_name if on Windows". Example: ./Car4/img: \n'))

        rect = [75, 140, 70, 110]   #start = 20
        frames, template, rect = runVidOp(directory, start=15, rect=rect)
        fr.LKRegisteration(frames, template, rect, rotate=1, robust=True, numberOfiteration=1000, delta_p_threshold=0.1)  # rect update
        #Robust version
        # rect = [70, 170, 58, 145]   #start = 15
        # frames, template, rect = runVidOp(directory, start=15, rect=rect)
        # fr.LKRegisteration(frames, template, rect, rotate=-1, robust=True, numberOfiteration=600, delta_p_threshold=0.11) # rect update

        print("Problem 2 finished")

    if problem == 3:
        directory = './DragonBaby/img'
        directory = str(input('What is the name and directory of the DragonBaby folder with the images? Note, this should be entered as"./folder_name if on Windows" Example: ./DragonBaby/img: \n'))
        # rect = [150, 220, 75, 150]   # left, right, up, bottom bound
        # rect = [160, 250, 75, 150]
        rect = [135, 220, 70, 195]
        frames, template, rect = runVidOp(directory, start=0, rect=rect)
        fr.LKRegisteration(frames, template, rect, rotate=2, robust=False, numberOfiteration=4000, delta_p_threshold=0.15)  # rect update
        print("Problem 3 finished")
    else:
        Exception("No such problem")
        AssertionError("No such problem")
        print("No such problem")
    prgRun = False
    return prgRun


print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()
