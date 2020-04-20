import numpy as np
from __main__ import *
from HistorgramOfLanePixels import *


def fitThemLines(x, y, n,allpoints):

    lines=np.polyfit(y,x, 2,True)
    lines=np.poly1d(lines)



    xpoints=lines(allpoints)



    lines=np.array([xpoints,allpoints],np.int32).T



    return lines, xpoints


def MarkLanes(bincntframe,flatBGR,frame):

    Tweight=0
    Mweight=0
    Bweight=0
    bincntframeTop = bincntframe[:int(bincntframe.shape[0] / 3), :]
    bincntframeMid = bincntframe[int(bincntframe.shape[0] / 3):int(bincntframe.shape[0] * 2 / 3), :]
    bincntframeBot = bincntframe[int(bincntframe.shape[0] * 2 / 3):, :]

    ypoints = np.array([int((0 + int(bincntframe.shape[0] / 3)) / 2),
                        int((int(bincntframe.shape[0] / 3) + int(bincntframe.shape[0] * 2 / 3)) / 2),
                        int((int(bincntframe.shape[0] * 2 / 3) + bincntframe.shape[0]) / 2)]).T

    LeftT, RightT = findLeftAndRightPoints(bincntframeTop)
    LeftM, RightM = findLeftAndRightPoints(bincntframeMid)
    LeftB, RightB = findLeftAndRightPoints(bincntframeBot)
    ######################################################################################################

    bincntframeTH = bincntframe[:int(bincntframe.shape[0] / 2), :]
    bincntframeBH = bincntframe[int(bincntframe.shape[0] / 2):, :]
    LeftTH, RightTH = findLeftAndRightPoints(bincntframeTH)
    LeftBH, RightBH = findLeftAndRightPoints(bincntframeBH)
    ######################################################################################################



    LeftTotal, RightTotal = findLeftAndRightPoints(bincntframe)
    ######################################################################################################


    Xright = np.array([RightTotal, RightTotal, RightTotal])
    Xleft = np.array([LeftTotal, LeftTotal, LeftTotal])
    # ypoints = np.array([int((0 + int(bincntframe.shape[0]/2)) / 2),                   int((int(bincntframe.shape[0]) + bincntframe.shape[0]) / 2)]).T
    ######################################################################################################



    # Xright = np.array([RightTH, RightBH])
    # Xleft = np.array([LeftTH, LeftBH])



    wholeframe = np.arange(0, frame.shape[1])

    Leftlines, LeftDiferential = fitThemLines(Xleft, ypoints, 3, wholeframe)

    Rightlines, RightDiferential = fitThemLines(Xright, ypoints, 3, wholeframe)

    Turning =(RightDiferential-LeftDiferential)

    LeftlinesM=Leftlines[200:1100]
    RightlinesM = Rightlines[200:1100]

    flatBGR = cv2.polylines(flatBGR, [LeftlinesM], False, (0, 0, 255), 5)
    MarkedLanes = cv2.polylines(flatBGR, [RightlinesM], False, (255, 0, 0), 5)


    return MarkedLanes,Leftlines,Rightlines, Turning


def CheckTurn(Right,Left,frame):
    RightX=Right[:,0]
    LeftX=Left[:,0]
    RightY = Right[:, 1]
    LeftY = Left[:, 1]

    wholeframe = np.arange(0, frame.shape[1])


    LeftBotX = LeftX[len(LeftX) - 200]
    RightBotX = RightX[len(RightX) - 200]
    LeftTopX = LeftX[0]
    RightTopX = RightX[0]

    LeftBotY = LeftY[len(LeftY) - 200]
    RightBotY = RightY[len(RightY) - 200]
    LeftTopY = LeftY[0]
    RightTopY = RightY[0]



    MidTopX=int((LeftTopX+RightTopX)/2)
    MidBotX=int((LeftBotX+RightBotX)/2)

    MidTopY = int((LeftTopY + RightTopY) / 2)
    MidBotY = int((LeftBotY + RightBotY) / 2)

    MidX=np.array([MidTopX,MidBotX])
    MidY=np.array([MidTopY,MidBotY])

    MidLine, MidDifferntial = fitThemLines(MidX, MidY, 3, wholeframe)
    Gradiantset=np.gradient(MidLine[:,0])
    Gradiant=np.abs(np.sum(Gradiantset,axis=0))

    if Gradiant <len(Gradiantset):
        print('Turning Right')
    elif Gradiant >=len(Gradiantset):
        print('Turning Left')


    return MidLine