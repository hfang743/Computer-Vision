import numpy as np
import perspective
import cv2
import imutils


def homoToResCenter(src_corners_coordinates, dest_corners_coordinates):
    homo, _ = cv2.findHomography(src_corners_coordinates, dest_corners_coordinates, 0)
    print(homo)
    return homo


def eyeballLaneCorners(img_gray, centers_coordinates):
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for center_coordinates in centers_coordinates:
        img_gray = cv2.circle(img_gray, center_coordinates, 5, [0, 255, 0], -1)

    cv2.imshow('Image with corners', img_gray)
    cv2.waitKey(0)
    return np.asarray(centers_coordinates)


def translation(xy, dest_corners_coordinates):
    dest_corners_coordinates, _ = perspective.shiftToASpot(dest_corners_coordinates, xy)


def debug():
    """
    first show the whole process of manually adjust the corners from the lane used to compute homo
    second show the resultant unwarp image"""
    data_choice = 1
    img_path = None
    corners_warped = None
    corners_unwarped = None

    """read the image"""
    if data_choice == 1:
        img_path = "./data/0000000000.png"
    elif data_choice == 2:
        img_path = "./challenge_video_cache/challenge_video_Moment2.jpg"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """adjust the corners of lane on target image"""
    if data_choice == 1:
        corners_warped = [(512, 325), (760, 325), (886, 470), (274, 470)]  # four corners on lane
        width = 108
        length = 400  # longer side of rectangle
        corners_unwarped = perspective.rectangle(width, length)  # width and height of rectangle (108, 400)
        offset = np.array([img_gray.shape[0] / 2, img_gray.shape[1] - length * (17 / 12)]).astype(
            int)  # where the rectangle to be on dest image
        corners_unwarped, _ = perspective.shiftToASpot(corners_unwarped, offset)
    elif data_choice == 2:
        corners_warped = [(598, 495), (758, 495), (876, 592), (462, 592)]
        width = 108
        length = 400  # longer side of rectangle
        corners_unwarped = perspective.rectangle(width, length)
        offset = np.array([img_gray.shape[0] / 2, img_gray.shape[1] - length * (4 / 3)]).astype(
            int)  # where the rectangle to be on dest image
        corners_unwarped, _ = perspective.shiftToASpot(corners_unwarped, offset)

    """eyeball the corner of lane for images input"""
    corners_warped = eyeballLaneCorners(img_gray, corners_warped)
    """calculate the homograph"""
    homo = homoToResCenter(corners_warped, corners_unwarped)
    img_unwarped = cv2.warpPerspective(img, homo, (img_gray.shape[0], img_gray.shape[1]))

    # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Image unwarped', img_unwarped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def homo_problem2():
    return np.array([[-1.90187029e-01, -1.20514938e+00, 4.00830702e+02],
                     [5.36923043e-16, -5.73597280e+00, 1.59123755e+03],
                     [5.80470942e-19, -4.42069468e-03, 1.00000000e+00]])


def homo_problem3():
    return np.array([[-9.50548952e-02, -8.38508302e-01, 4.28812884e+02],
                     [2.57889823e-17, -2.76094728e+00, 1.28978006e+03],
                     [3.00026378e-20, -2.30469104e-03, 1.00000000e+00]])

# debug()
