import numpy as np
import imagesProcessing as ip

import cv2

def checkRange():
    a = np.zeros((5, 8))
    point_upperLeft = np.array([1, 0])
    point_lowerRight = np.array([7, 2])
    width = int(point_lowerRight[0] - point_upperLeft[0])  # width of bounding box
    height = int(point_lowerRight[1] - point_upperLeft[1])  # height of bounding box
    template = a[point_upperLeft[1]:point_lowerRight[1]+1,point_upperLeft[0]:point_lowerRight[0]+1]

    Jacobian_W = np.zeros(((width+1) * (height+1), 3))  # initialize the Jacobian of affine transformation matrix
    index = 0
    for y in range(point_upperLeft[1], point_lowerRight[1]+1):    # loop over range of y: y is second element in a variable vector
        for x in range(point_upperLeft[0], point_lowerRight[0]+1):    # loop over range of x: x is first element in a variable vector
            Jacobian_W[index, :] = np.array([x, y, index])
            index = index + 1

    print(a)
    print(Jacobian_W)
    print(Jacobian_W.shape)
    print(template.shape)


def checkRange1():
    rect = np.array([[1, 2],[0, 2],[1, 1]])
    rects = ip.rectangleToCoodinatesArray(rect)
    print(rects)
    print(rects.shape)


def checAffineWarp():
    frame_orig = cv2.imread('./Bolt2/img/0001.jpg')
    affine = np.array([[1, 0, 0],[0, 1, 0]]).astype(float)
    print(affine.shape)
    frame_move = cv2.warpAffine(frame_orig, affine, (frame_orig.shape[1], frame_orig.shape[0]))

    cv2.imshow("Before move", frame_orig)
    cv2.imshow("After move", frame_move)
    cv2.waitKey(0)

    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

def compare():
    template = cv2.imread('./Car4/img/template.jpg')
