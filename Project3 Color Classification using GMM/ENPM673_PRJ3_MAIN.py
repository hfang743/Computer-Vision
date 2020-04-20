import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils
import os
import random

from getData import *
from histogramImages import *
from em_Nick import *

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True



def main(prgRun):
    getdata = False
    Showflag = False
    SaveFlag = False

    getdataYN = str.lower(input('Create a data set? Enter |yes| or |no|: '))
    if getdataYN=='yes':
        getdata = True

    ShowflagYN = str.lower(input('Show histograms? Enter |yes| or |no|: '))
    if ShowflagYN == 'yes':
        Showflag = True

    SaveFlagYN = str.lower(input('Save off data set and train? Enter |yes| or |no|: '))
    if SaveFlagYN == 'yes':
        SaveFlag = True


    LearnFlag = SaveFlag

    Ohb, Ohg, Ohr = orangeHist(Showflag, SaveFlag)
    Ghb, Ghg, Ghr = greenHist(Showflag, SaveFlag)
    Yhb, Yhg, Yhr = yellowHist(Showflag, SaveFlag)

    video = cv2.VideoCapture('detectbuoy.avi')

    # Read until video is completed
    if getdata == True:
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            if ret == True:
                # frame = imutils.resize(frame, width=320, height=180)
                frame.shape
                ogframe = frame
                clnframe = frame
                resetframe = frame
                # cv2.imshow('Original Frame', frame)
                # if cv2.waitKey(25) & 0xFF == ord('q'):
                #     break

                buildData(frame)
    else:
        em_NickMain(LearnFlag)


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
