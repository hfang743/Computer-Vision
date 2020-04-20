import cv2
import os
import numpy as np


def firstTemplateAndSecondFrameAndImageList(path, flag_covertToGray=True):
    images = []
    cv_img = []
    for image in os.listdir(path):
        images.append(image)
    images.sort()
    for image in images:
        img = cv2.imread("%s/%s" % (path, image))
        if flag_covertToGray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_img.append(img)
    """get rectangle and template"""
    return cv_img


def rectangleBoundingBoxToInternalPxielsCoodinatesArray(rect):
    """
        Parameters
        ----------
        rect : np.ndarray <float>
            a bounding box that marks the template region in previous frame, the rectangle coordinates that intend to define the
            tracking feature location at the image
        Returns
        ----------
        points_in_rect: : np.ndarray <float>
            all points in a bounding box line up in columns
    """
    assert type(rect) == np.ndarray
    assert rect.shape == (3, 2)

    point_upperLeft = rect[0:2, 0].astype(int)  # smaller
    point_lowerRight = rect[0:2, 1].astype(int)  # bigger
    width = int(point_lowerRight[0] - point_upperLeft[0])  # width of bounding box
    height = int(point_lowerRight[1] - point_upperLeft[1])  # height of bounding box
    points_in_rect_x = np.zeros((3, ((width + 1))))  # initialize the Jacobian of affine transformation # matrix
    """
    index = 0
    for y in range(point_upperLeft[1],
                   point_lowerRight[1] + 1):  # loop over range of y: y is second element in a variable vector
        for x in range(point_upperLeft[0],
                       point_lowerRight[0] + 1):  # loop over range of x: x is first element in a variable vector
            points_in_rect[:, index] = np.array([x, y, 1])
            index = index + 1
    """
    points_in_rect_x[0, :] = range(point_upperLeft[0], point_lowerRight[0] + 1)  # assign x
    points_in_rect_x[1, :] = point_upperLeft[1]
    points_in_rect = points_in_rect_x
    for y in range(point_upperLeft[1],
                   point_lowerRight[1]):  # loop over range of y: y is second element in a variable vector
        points_in_rect_x[1, :] = points_in_rect_x[1, :] + 1
        points_in_rect = np.concatenate((points_in_rect, points_in_rect_x), axis=1)
    assert points_in_rect.shape == (3, (width + 1) * (height + 1))
    return points_in_rect


def subImageInBoundingBoxAndEq(img, rect, histEqualize=False):
    assert type(img) == np.ndarray and type(rect) == np.ndarray
    assert rect.shape == (3, 3)
    point_upperLeft = rect[0:2, 0].astype(int)
    point_upperRight = rect[0:2, 1].astype(int)
    point_lowerRight = rect[0:2, 2].astype(int)
    img_cropped = img[point_upperLeft[1]:point_lowerRight[1] + 1, point_upperLeft[0]:point_lowerRight[0] + 1]
    assert img_cropped.shape[1] == point_lowerRight[0] - point_upperLeft[0] + 1, "img cut to size: " + str(
        img_cropped.shape[0]) + " in x, and supposed to be " + str(
        int(point_lowerRight[0] - point_upperLeft[0]))  # number of pixles in x smaller than range
    assert img_cropped.shape[0] == point_lowerRight[1] - point_upperLeft[1] + 1, "img cut to size: " + str(
        img_cropped.shape[1]) + " in y, and supposed to be " + str(
        int(point_lowerRight[1] - point_upperLeft[1]))  # number of pixles in y smaller than range
    # if histEqualize:
        # img_cropped = cv2.equalizeHist(img_cropped)
    return img_cropped


def corpSubImageIn3Points(img, rect):
    assert type(img) == np.ndarray and type(rect) == np.ndarray
    assert rect.shape == (3, 3)
    point_upperLeft = list(rect[0:2, 0].astype(int))  # shape is (2,)
    point_upperRight = list(rect[0:2, 1].astype(int))
    point_lowerRight = list(rect[0:2, 2].astype(int))
    point_lowerLeft = [point_upperLeft[0], point_lowerRight[1]]
    points = np.array([point_upperLeft, point_upperRight, point_lowerRight, point_lowerLeft])

    """source start: credit: Knight 金, https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from
    -image-using-opencv-python """
    # Crop the bounding rect
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    croped = img[y:y + h, x:x + w].copy()

    # make mask
    points = points - points.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    # do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    # add the white background
    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    img_cropped = bg + dst
    """source end"""
    return img_cropped


def drawRect(img, rect, flag_show=False, flag_hold=False):
    img_copy = img.copy()
    p1, p2, p3 = np.transpose(rect.astype(int))[:, 0:3]
    p3 = tuple(p3[0:2])
    p2 = tuple(p2[0:2])
    p1 = tuple(p1[0:2])
    cv2.rectangle(img_copy, p1, p3, (0, 255, 0))
    if flag_show:
        cv2.imshow("Image with marked feature", img_copy)
        if flag_hold:
            if cv2.waitKey(0):
                cv2.destroyAllWindows()
        else:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
    return img_copy


def fixRect(rect, rectTemp, prect):
    flag=0
    if np.any(rect<0):
        print('rect invert')
        rect = np.abs(rect)

    point_upperLeft = rect[0:2, 0].astype(int)
    point_lowerRight = rect[0:2, 2].astype(int)
    point_upperRight = rect[0:2, 1].astype(int)


    width = int(point_lowerRight[0] - point_upperLeft[0])  # width of bounding box
    height = int(point_lowerRight[1]-point_upperLeft[1])  # height of bounding box

    point_upperLeftT = rectTemp[0:2, 0].astype(int)
    point_lowerRightT = rectTemp[0:2, 2].astype(int)

    widthT = int(point_lowerRightT[0] - point_upperLeftT[0])  # width of bounding box
    heightT = int(point_lowerRightT[1]-point_upperLeftT[1])  # height of bounding box

    WR=np.absolute(width/widthT)
    HR=np.absolute(height/heightT)

    Limit=3


    if Limit>WR and Limit>HR:
        prect=rect

    return rect, prect, flag


def getPixelsValue(rects, transformation_warp):
    assert type(rects) == np.ndarray and type(transformation_warp) == np.ndarray
    assert rects.shape[1] == 3
    assert transformation_warp.shape == (3, 3)
    return np.dot(transformation_warp, rects)


def imgToArray(img):
    """change the (n x m) shape array to (n*m, 1) array"""
    assert type(img) == np.ndarray
    width, height = img.shape
    return img.reshape((width * height, 1))


def uint8ToFloat(img):
    """change the image intensity from uint8 to float"""
    img_copy = img.astype(float) / 255.0
    return img_copy


def floatToUint8(img):
    """change the image intensity from uint8 to float"""
    img_copy = img * 255.0
    return img_copy.astype(np.uint8)


def checkIntensity(img1, img2):
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    h1 = cv2.calcHist([img1], [0], None, [256], (0, 256))
    h2 = cv2.calcHist([img2], [0], None, [256], (0, 256))
    return cv2.compareHist(h1, h2, method=cv2.HISTCMP_CORREL)
