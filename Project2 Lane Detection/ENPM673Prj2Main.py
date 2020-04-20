import numpy as np
import cv2
import imutils
import math

from imageFileNames import imagefiles
from imagecorrection import *
import HomoCalculation
from HistorgramOfLanePixels import *
from fitlines import *
import random
import perspective
from regionOfInterest import *
import warnings
warnings.simplefilter('ignore', np.RankWarning)

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True

def main(prgRun):
    problem = 3
    problem = int(input('Which part would you like to run? \nEnter 1 for ngihtime image . \nEnter 2 for the firt part of the lane finder. \nEnter 3 for the second part of the lane finder (Challenge video): '))

    #Correct image
    if problem ==1:

        video = cv2.VideoCapture('Night Drive - 2689.mp4')

        # Read until video is completed
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            if ret == True:
                frame = imutils.resize(frame, width=320, height=180)
                frame.shape
                ogframe = frame
                clnframe = frame
                resetframe = frame
                cv2.imshow('Original Frame', frame)

            ##########################Correct frame###########################
            gamma=5
            sat=5
            hue=1
            contrast = 60

            frame = adjust_gamma(frame, gamma)
            frame = adjustSaturation(frame, hue, sat)
            frame=adjustContrast(frame, contrast)
            frame = cv2.bilateralFilter(frame, 15, 75, 75)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

            ###################Output Imagery##########################
            cv2.imshow('DWF', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


    #Lane finder Image set
    elif problem == 2:
        directory = './data'
        directory=str(input('What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows": \n'))

        print("Getting images from " + str(directory))
        imageList = imagefiles(directory)  # get a stack of images
        homo = HomoCalculation.homo_problem2()
        homo_inv = perspective.invHomo(homo)
        """process each image individually"""
        for i in range(len(imageList)):
            frameDir = directory + '/' + imageList[i]
            frame=cv2.imread(frameDir)
            # frame = imutils.resize(frame, width=320, height=180)

            ##########################Correct frame###########################
            K, D = cameraParamsPt2()
            frame = cv2.undistort(frame, K, D, None, K)
            ##########################Thresh frame###########################
            grayframe=grayscale(frame)
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            binaryframe = yellowAndWhite(frame)
            ############################Region ################
            region = process_image(binaryframe)
            #####################Homography and dewarp########################

            """the next line give you a flat view of current frame"""
            flatfieldBinary = cv2.warpPerspective(region, homo, (frame.shape[0], frame.shape[1]))
            flatBGR= cv2.warpPerspective(frame, homo, (frame.shape[0], frame.shape[1]))


            ####################Contour#######################################
            cnts, hierarchy = cv2.findContours(flatfieldBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            bincntframe = cv2.drawContours(flatfieldBinary, cnts, -5, 255, 5)
            # cv2.imshow('flatfield binary', bincntframe)

            # cntframe = cv2.drawContours(flatBGR, cnts,-5, (255, 0, 0), 5)

            ###################Draw Lines##########################################
            LanesDrawn, LeftLines, RightLines, Turning = MarkLanes(bincntframe, flatBGR, frame)
            # cv2.imshow('Flafield Lanes', LanesDrawn)

            leftLane_warped = perspective.perspectiveTransfer_coord(LeftLines, homo_inv)[250:1200]
            rightLane_warped = perspective.perspectiveTransfer_coord(RightLines, homo_inv)[250:1200]

            Turning = CheckTurn(rightLane_warped, leftLane_warped, frame)[300:1000]

            MidLane = cv2.polylines(frame, [Turning], False, (0, 255, 0), 5)
            frame_lane = cv2.polylines(MidLane, [leftLane_warped], False, (0, 0, 255), 5)
            frame_lane = cv2.polylines(frame_lane, [rightLane_warped], False, (255, 0, 0), 5)

            ###################Output Imagery##########################
            # frame_lane = imutils.resize(frame_lane, width=320, height=180)
            # LanesDrawn = imutils.resize(LanesDrawn, width=320, height=180)

            cv2.imshow('Working Frame', LanesDrawn)
            cv2.imshow("Imposed Lane Lines", frame_lane)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            if flag:
                cv2.imshow('unwarped video', img_unwarped)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break



    #Lane Finder Challenge vid
    elif problem == 3:

        video = cv2.VideoCapture('challenge_video.mp4')
        homo = HomoCalculation.homo_problem3()
        homo_inv = perspective.invHomo(homo)
        # Read until video is completed
        while (video.isOpened()):
            # Capture frame-by-frame
            ret, frame = video.read()
            if ret == True:
                # frame = imutils.resize(frame, width=320, height=180)
                # frame = frame[int(frame.shape[0] / 2) + 70:, :]
                ogframe = frame
                clnframe = frame
                resetframe = frame

                ##########################Correct frame###########################
                K,D=cameraParamsPt3()
                frame = cv2.undistort(frame, K, D, None, K)

                ##########################Thresh frame###########################
                grayframe = grayscale(frame)
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
                binaryframe = yellowAndWhite(frame)
                ############################Region ################
                region = process_image(binaryframe)
                #####################Homography and dewarp########################

                #check

                """the next line give you a flat view of current frame"""
                flatfieldBinary = cv2.warpPerspective(region, homo, (frame.shape[0], frame.shape[1]))
                flatBGR = cv2.warpPerspective(frame, homo, (frame.shape[0], frame.shape[1]))

                ####################Contour#######################################
                cnts, hierarchy = cv2.findContours(flatfieldBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                bincntframe = cv2.drawContours(flatfieldBinary, cnts, -5, (255), 5)
                # cv2.imshow('flatfield binary', bincntframe)

                # cntframe = cv2.drawContours(flatBGR, cnts,-5, (255, 0, 0), 5)

                ###################Draw Lines##########################################


                LanesDrawn, LeftLines, RightLines, null = MarkLanes(bincntframe, flatBGR, frame)

                leftLane_warped = perspective.perspectiveTransfer_coord(LeftLines, homo_inv)[100:1100]
                rightLane_warped = perspective.perspectiveTransfer_coord(RightLines, homo_inv)[100:1100]

                Turning=CheckTurn(rightLane_warped,leftLane_warped,frame)[500:1000]

                MidLane = cv2.polylines(frame, [Turning], False, (0, 255, 0), 5)
                frame_lane = cv2.polylines(MidLane, [leftLane_warped], False, (0, 0, 255), 5)
                frame_lane = cv2.polylines(frame_lane, [rightLane_warped], False, (255, 0, 0), 5)

                ###################Output Imagery##########################
                LanesDrawn = imutils.resize(LanesDrawn, width=320, height=180)
                cv2.imshow('Working Frame', LanesDrawn)

                frame_lane = imutils.resize(frame_lane, width=320, height=180)
                cv2.imshow('Final Frame', frame_lane)


                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()
    prgRun=False
    return prgRun







print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()