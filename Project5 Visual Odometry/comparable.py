import matplotlib.pyplot as plt
import os
from Oxford_dataset.ReadCameraModel import *
from Oxford_dataset.UndistortImage import *
from imageCorrection import *
import cv2
import numpy as np
import random

l = []
frames1 = []


def getcomparablePoints(distort1, distort2, sift):
    kp1, des1 = sift.detectAndCompute(distort1, None)
    kp2, des2 = sift.detectAndCompute(distort2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    pointsmatched = []
    pointsfrom1 = []
    pointsfrom2 = []

    for i, (s, p) in enumerate(matches):
        if s.distance < 1 * p.distance:
            pointsmatched.append(s)
            pointsfrom2.append(kp2[s.trainIdx].pt)
            pointsfrom1.append(kp1[s.queryIdx].pt)

    pointsfrom1 = np.float32(pointsfrom1)
    pointsfrom2 = np.float32(pointsfrom2)

    return pointsfrom1, pointsfrom2


def getcomparableF(pointsfrom1, pointsfrom2):
    F, mask = cv2.findFundamentalMat(pointsfrom1, pointsfrom2, cv2.FM_RANSAC)

    pointsfrom1 = pointsfrom1[mask.ravel() == 1]
    pointsfrom2 = pointsfrom2[mask.ravel() == 1]

    return F, pointsfrom1, pointsfrom2


def getcomparableRT(F, k, pointsfrom1, pointsfrom2):
    E = np.matmul(k.T, np.matmul(F, k))
    retval, R, t, mask = cv2.recoverPose(E, pointsfrom1, pointsfrom2, k)
    return R, t


def cameraMatrix(file):
    frames1 = []
    for frames in os.listdir(file):
        frames1.append(frames)
    fx, fy, cx, cy, G_camera_frames, LUT = ReadCameraModel('Oxford_dataset/model/')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K, LUT


def undistortImg(img):
    colorimage = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistortedimage = UndistortImage(colorimage, LUT)
    gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)

    return gray


def getcomprableHomo(R, t):
    h = np.column_stack((R, t))
    a = np.array([0, 0, 0, 1])
    h = np.vstack((h, a))
    return h

# sift = cv2.xfeatures2d.SIFT_create()
# file = "Oxford_dataset/stereo/centre/"
# k, LUT = cameraMatrix(file)
#
# for frames in os.listdir(file):
#     frames1.append(frames)
#
# homo1 = np.identity(4)
# t1 = np.array([[0, 0, 0, 1]])
# t1 = t1.T
#
# for index in range(19, 1700):  # len(frames1) - 1
#     img1 = cv2.imread("%s%s" % (file, frames1[index]), 0)
#     distort1 = undistortImg(img1)
#
#     img2 = cv2.imread("%s%s" % (file, frames1[index + 1]), 0)
#     distort2 = undistortImg(img2)
#
#     R, T = poseMatrix(distort1, distort2, k)
#
#     homo2 = Homogenousmatrix(R, T)
#     homo1 = np.matmul(homo1, homo2)
#     p1 = np.matmul(homo1, t1)
#
#     plt.scatter(p1[0][0], -p1[2][0], color='r')
#     l.append([p1[0][0], -p1[2][0]])
#
#     if index%10==0:
#         print(index)
#
# saveVar(l, 'visited points')
#
#
# for index in range(19,1700):  # len(frames1) - 1 if read all frames
#     img1 = cv2.imread("%s%s" % (file, frames1[index]), 0)
#
#     plt.scatter(l[index-19][0], l[index-19][1], color='r')
#
#     plt.pause(0.00001)
#
#     cv2.imshow('img1', img1)
#     if cv2.waitKey(2) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()

