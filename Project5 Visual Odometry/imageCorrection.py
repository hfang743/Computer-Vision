from __main__ import *
import pickle
import numpy as np
import cv2
from imageFileNames import *


def saveVar(var,varprint):
    with open(varprint, 'wb') as f:
        pickle.dump(var, f)

def openVar(varprint):
    with open(varprint, 'rb') as f:
        var = pickle.load(f)
    return var

def correctImages(directory, LUT):
    print('WARNING: THIS WILL TAKE A WHILE')
    imageList = imagefiles(directory)
    frameset = []
    for i in range(len(imageList)):
        if i % 100 == 0:
            print(i, '/', len(imageList))
        frameDir = directory + '/' + imageList[i]
        frame = cv2.imread(frameDir, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BayerGR2BGR)
        frame = UndistortImage(frame, LUT)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameset.append(frame)
    print("Done correcting Images")
    saveVar(frameset, 'CorrectedFrames')