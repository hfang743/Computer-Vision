import cv2
import numpy as np
import math
import imutils
import perspective
import findAngle
from __main__ import *

print('Imports Complete')
print('CV2 version')
print(cv2.__version__)

flag_debug = False
side_AR_Tag_estimation = 150
intrinsic_camera_matrix = np.array([[1406.08415449821, 0, 0],
                                    [2.20679787308599, 1417.99930662800, 0],
                                    [1014.13643417416, 566.347754321696, 1]]).T
Number_grid = 8


def setUpBolbDetector():
    """set blob detector"""
    detector = cv2.SimpleBlobDetector_create()
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 22000
    params.maxArea = 400000
    params.filterByColor = True
    params.blobColor = 255
    params.filterByConvexity = True
    params.minConvexity = 0.88
    detector = cv2.SimpleBlobDetector_create(params)
    return detector


def loadImages_refMarker():
    """input reference marker images"""
    global AR_tag_reference
    AR_tag_reference = cv2.imread("./data/ref_marker.png")


def compress(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def cutInterestingAreaFromFrame(AR_tag_potential, res):
    (interestingArea_center_x, interestingArea_center_y) = AR_tag_potential.pt  # center of key points
    interestingArea_center_x = int(interestingArea_center_x)
    interestingArea_center_y = int(interestingArea_center_y)
    interestingArea_radius = int(np.ceil(AR_tag_potential.size))  # radius of key points, reformat to integer
    """Create a image contains a AR tag from the frame, and find all contours in this interesting area"""
    img_gray_AR_tags_fromBackGround = res[(interestingArea_center_y - interestingArea_radius):(
            interestingArea_center_y + interestingArea_radius),
                                      (interestingArea_center_x - interestingArea_radius):(
                                              interestingArea_center_x + interestingArea_radius)]
    return interestingArea_center_x, interestingArea_center_y, interestingArea_radius, img_gray_AR_tags_fromBackGround


def cube_top(perjectionMatrix, corners_inFrame, corners_flatView):
    """perjection Matrix is the perjection matrix from 3D to 2D
    corners_inFrame is a stack of points in frame view
    corners_flatView is a stack of points in flat view"""
    perjectionMatrix = perjectionMatrix / perjectionMatrix[2, 3]
    """make a top square in 3D homogeneous"""
    side = side_AR_Tag_estimation
    topCorners_flatView_3D_homogeneous = np.zeros([corners_flatView.shape[0], 4])
    topCorners_flatView_3D_homogeneous[:, 0] = corners_flatView[:, 0]
    topCorners_flatView_3D_homogeneous[:, 1] = corners_flatView[:, 1]
    topCorners_flatView_3D_homogeneous[:, 2] = topCorners_flatView_3D_homogeneous[:, 2] - side
    topCorners_flatView_3D_homogeneous[:, 3] = topCorners_flatView_3D_homogeneous[:, 3] + 1
    topCorners_flatView_3D_homogeneous = topCorners_flatView_3D_homogeneous.transpose()
    """cast the top corner to frame view"""
    topCorners_flatView_2D_homogeneous = np.dot(perjectionMatrix, topCorners_flatView_3D_homogeneous)
    topCorners_flatView_2D_homogeneous = topCorners_flatView_2D_homogeneous / topCorners_flatView_2D_homogeneous[2, :]
    topCorners_flatView_2D_homogeneous = topCorners_flatView_2D_homogeneous.transpose()
    topCorners_flatView_2D_homogeneous = topCorners_flatView_2D_homogeneous[:, 0:2]
    return topCorners_flatView_2D_homogeneous.astype(int)


def draw_cube(perjectionMatrix, corners_inFrame, corners_flatView, img):
    """
    perjection Matrix is the perjection matrix from 3D to 2D
    corners_inFrame is a stack of points in frame view
    corners_flatView is a stack of points in flat view
    im is the frame
    """
    topCorners_inFrame = cube_top(perjectionMatrix, corners_inFrame, corners_flatView)
    """draw the base (Blue)"""
    img = cv2.drawContours(img, [corners_inFrame], -1, (0, 255, 0), 3)
    """draw the column (Green)"""
    # draw pillars in blue color
    for i, j in zip(range(4), range(4)):
        img = cv2.line(img, tuple(corners_inFrame[i]), tuple(topCorners_inFrame[j]), (255, 0, 0), 3)
    """draw the top(Red)"""
    img = cv2.drawContours(img, [topCorners_inFrame], -1, (0, 0, 255), 3)
    return img


def askUserInput():
    """ask user for command"""
    datachoice = int(input('\nWhich video data would you like to use? \nPlease enter 1, 2, or 3: '))
    inputNum = int(input('\nIdentify QR code? Impose Image? Impose Cube? \nPlease enter 1, 2, or 3: '))
    command = ["Identify QR code", "Impose Lena", "Impose Cube"]
    if datachoice != 1 and datachoice != 2 and datachoice != 3:
        print('End Program')
    return datachoice, inputNum, command[inputNum - 1]


def loadData(datachoice):
    """load data based on command input"""
    img_Lena = cv2.imread("./resource/Lena.png")
    video = None
    if datachoice == 1:
        video = cv2.VideoCapture('./resource/data_1.mp4')
    if datachoice == 2:
        video = cv2.VideoCapture('./resource/data_2.mp4')
    if datachoice == 3:
        video = cv2.VideoCapture('./resource/data_3.mp4')
    assert video != None, "Video loading fail"
    """get the frame resolutions"""
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    print("Video frame size is " + str(frame_width) + ", " + str(frame_height))
    """resize Lena"""
    img_Lena = compress(img_Lena, 50)
    print("Lena image size is " + str(img_Lena.shape[0]) + ", " + str(img_Lena.shape[1]))
    return video, frame_width, frame_height, img_Lena


def main():
    """This is main function of Project 1"""
    """Here is some constant and control variables"""
    videoOuput_flag = False
    KeyPoints_flag = False
    Decode_flag = False
    Lena_flag = False
    Cube_flag = False

    datachoice, section, command = askUserInput()    # ask user what to do
    # datachoice = 1
    # section = 1
    # command = ""
    """input images or video"""
    cap, frame_width, frame_height, Lena = loadData(datachoice)
    """output images or video"""
    if videoOuput_flag: out = cv2.VideoWriter("output_data"+str(datachoice)+" "+command+".avi", -1, 20.0, (frame_width, frame_height))
    """set up blob detector"""
    detector = setUpBolbDetector()
    print('Detector initializations complete')
    """process the video based on the command input"""
    if section == 1:
        Decode_flag = True
    elif section == 2:
        Lena_flag = True
    elif section == 3:
        Cube_flag = True
    else:
        print("Wrong input command")
    """open the video, deal each frame"""
    while cap.isOpened():
        suc, frame = cap.read()
        img_outPut = frame  # a copy of colorful frame for output, and any output element will be added to this image
        """Pre process a frame"""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to gray scalar
        """filter all the pixels below 200"""
        lowerBoundary = 200
        upperBoundary = 255
        mask = cv2.inRange(frame_gray, lowerBoundary, upperBoundary)  # get between 180 and 255 range
        res = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)  # make the gray image to be float32 format
        AR_tags_potential, AR_Tag_draw = findAngle.findARTagFromFrame(frame_gray, detector, mask,
                                                            res)  # a list that store all possible area that may contain AR tag
        # print("There may be " + str(len(AR_tags_potential)) + " AR tags")
        for i in range(0, len(AR_tags_potential)):
            # cut a image from frame, this image contains a AR tag
            AR_Tag_center_x, AR_Tag_center_y, AR_Tag_center_radius, img_gray_AR_tags_fromBackGround = cutInterestingAreaFromFrame(
                AR_tags_potential[i], res)
            # three basic info from the cut image (a AR tag image)
            # print("     #" + str(i) + " size:" + str(AR_Tag_center_radius))
            corners_outer, contour_outer, contour_inner = findAngle.findARTagContours(img_gray_AR_tags_fromBackGround)
            """if we could cut a possible AR tag from frame, the we keep going"""
            if contour_outer is not None:
                """change the coordinates from interesting area to frame"""
                AR_Tag_center_InFrame = np.array([AR_Tag_center_x, AR_Tag_center_y])  # - AR_Tag_center_radius
                corners_outer_inFrame, shouldBeCenter = perspective.shiftToASpot(corners_outer,
                                                                     AR_Tag_center_InFrame)  # corner points array, in original video frame coordinates
                corners_outer_inFrame = corners_outer_inFrame.astype(int)
                if flag_debug:
                    contour_inner_inFrame, shouldBeCenter = perspective.shiftToASpot(contour_inner,
                                                                         AR_Tag_center_InFrame)  # corner points array, in original video frame coordinates
                """figure out the homograph transformation from the a AR tag in frame to the AR tag in database"""
                if Decode_flag:
                    homo_FrameToFlat = perspective.Estimated_Homography(corners_outer,
                                                                        perspective.square(side_AR_Tag_estimation))
                    """Get a image that contains only AR tag after perspective transform"""
                    img_gray_AR_tags_fromBackGround_flat = perspective.perspectiveTransfer_image(
                        img_gray_AR_tags_fromBackGround, homo_FrameToFlat ,(side_AR_Tag_estimation, side_AR_Tag_estimation))
                    """compute the outter and inner contours and then switch back to 2D"""
                    contour_outer_transformed = perspective.perspectiveTransfer_coord(contour_outer, homo_FrameToFlat)
                    contour_inner_transformed = perspective.perspectiveTransfer_coord(contour_inner, homo_FrameToFlat)
                    if flag_debug:
                        """show the zoomed AR tag after perspective transform"""
                        # cv2.imshow("AR tag and contour draw in flatten image", img_gray_AR_tags_fromBackGround_flat)
                        # if cv2.waitKey(0) & 0xff == 27:
                        #     cv2.destroyAllWindows()
                        """show the contour of outer layer in zoomed image"""
                        img_GBR_AR_tags_fromBackGround_flat = cv2.cvtColor(img_gray_AR_tags_fromBackGround_flat,
                                                                           cv2.COLOR_GRAY2BGR)
                        img_GBR_AR_tags_fromBackGround_flat = cv2.drawContours(img_GBR_AR_tags_fromBackGround_flat,
                                                                               [contour_outer_transformed,
                                                                                contour_inner_transformed], -1,
                                                                               (0, 0, 255), 3)
                        """show those images"""
                        cv2.imshow("AR tag and contour draw in flatten image", img_GBR_AR_tags_fromBackGround_flat)
                        if cv2.waitKey(0) & 0xff == 27:
                            cv2.destroyAllWindows()
                    """find the direction, ID,  of AR tag by using the inner contour"""
                    angle, ID = findAngle.findAngleAndID(img_gray_AR_tags_fromBackGround_flat, contour_outer_transformed,
                                               contour_inner_transformed, Number_grid)
                if Lena_flag:
                    """get lena image processed"""
                    lena_size = Lena.shape[0]
                    # """get a homograph that takes ar tag to flat view"""
                    # homo_FrameToFlat = perspective.Estimated_Homography(corners_outer,
                    #                                                     perspective.square(side_AR_Tag_estimation))
                    # """Get a image that contains only AR tag after perspective transform"""
                    # img_gray_AR_tags_fromBackGround_flat = perspective.perspectiveTransfer_image(
                    #     img_gray_AR_tags_fromBackGround, side_AR_Tag_estimation, homo_FrameToFlat)
                    # """compute the outter and inner contours and then switch back to 2D"""
                    # contour_outer_transformed = perspective.perspectiveTransfer_coord(contour_outer, homo_FrameToFlat)
                    # contour_inner_transformed = perspective.perspectiveTransfer_coord(contour_inner, homo_FrameToFlat)
                    # if flag_debug:
                    #     """show the zoomed AR tag after perspective transform"""
                    #     # cv2.imshow("AR tag and contour draw in flatten image", img_gray_AR_tags_fromBackGround_flat)
                    #     # if cv2.waitKey(0) & 0xff == 27:
                    #     #     cv2.destroyAllWindows()
                    #     """show the contour of outer layer in zoomed image"""
                    #     img_GBR_AR_tags_fromBackGround_flat = cv2.cvtColor(img_gray_AR_tags_fromBackGround_flat,
                    #                                                        cv2.COLOR_GRAY2BGR)
                    #     img_GBR_AR_tags_fromBackGround_flat = cv2.drawContours(img_GBR_AR_tags_fromBackGround_flat,
                    #                                                            [contour_outer_transformed,
                    #                                                             contour_inner_transformed], -1,
                    #                                                            (0, 0, 255), 3)
                    #     """show those images"""
                    #     cv2.imshow("AR tag and contour draw in flatten image", img_GBR_AR_tags_fromBackGround_flat)
                    #     if cv2.waitKey(0) & 0xff == 27:
                    #         cv2.destroyAllWindows()
                    # """find the direction, ID,  of AR tag by using the inner contour"""
                    # angle, ID = findAngleAndID(img_gray_AR_tags_fromBackGround_flat, contour_outer_transformed,
                    #                            contour_inner_transformed)
                    """homograph that cast Lena to AR tag after perspective transform"""
                    homo_FrameToLena = perspective.Estimated_Homography(perspective.square(lena_size), corners_outer_inFrame)
                    """find AR tag angle"""
                    img_outPut = perspective.superimpose(img_outPut, Lena, homo_FrameToLena)
                    if flag_debug:
                        cv2.imshow("Lena after superimposing", img_outPut)
                        if cv2.waitKey(0) & 0xff == 27:
                            cv2.destroyAllWindows()
                if Cube_flag:
                    """draw cube"""
                    homo_FlattoFrame = perspective.Estimated_Homography(perspective.square(side_AR_Tag_estimation),
                                                                        corners_outer_inFrame)
                    pMatrix = perspective.projectionMatrix(intrinsic_camera_matrix, homo_FlattoFrame)
                    img_outPut = draw_cube(pMatrix, corners_outer_inFrame, perspective.square(side_AR_Tag_estimation),
                                           img_outPut)
                """mark the AR Tag first"""
                text = "AR Tag #" + str(i)
                if Decode_flag: text += (" ID is " + str(ID))
                img_outPut = cv2.putText(img_outPut, text, (int(AR_Tag_center_x + AR_Tag_center_radius/3), int(AR_Tag_center_y + AR_Tag_center_radius/3)) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        """process and output the resultant frame"""
        # write the flipped frame
        if videoOuput_flag:  out.write(img_outPut)
        if KeyPoints_flag: cv2.imshow("feature choice from original frame", AR_Tag_draw)
        cv2.imshow(str(command) + " resultant frame", img_outPut)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite("./output/" + str(command) + "_" + str(datachoice) + ".png", img_outPut)
    cap.release()
    if videoOuput_flag:  out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
