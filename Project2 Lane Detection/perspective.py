"""IMPORT"""
import numpy as np
import cv2


def square(side=200):
    """return four corner of square, the point vector is in row"""
    return np.array([[0, 0], [side, 0], [side, side], [0, side]])


def rectangle(short=200, long=400):
    """return four corner of square, the point vector is in row"""
    return np.array([[0, 0], [short, 0], [short, long], [0, long]])


def Estimated_Homography(p1, p2=square()):
    """Input two stack of points, each stack has four points vector in row,
    return the homography matrix from p1 to p2"""
    # check input points
    assert p1.shape == (4, 2), "P1 has size: " + str(p1.shape)
    assert p2.shape == (4, 2), "P1 has size: " + str(p2.shape)
    # assign values
    [x1, y1] = p1[0]
    [x2, y2] = p1[1]
    [x3, y3] = p1[2]
    [x4, y4] = p1[3]
    [xp1, yp1] = p2[0]
    [xp2, yp2] = p2[1]
    [xp3, yp3] = p2[2]
    [xp4, yp4] = p2[3]
    """
    x2 = 150
    x3 = 15
    x4 = 5
    y1 = 5
    y2 = 5
    y3 = 150
    y4 = 150
    xp1 = 100
    xp2 = 200
    xp3 = 220
    xp4 = 100
    yp1 = 100
    yp2 = 80
    yp3 = 80
    yp4 = 200
    """
    A = -np.array([
        [-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
        [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
        [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
        [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
        [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
        [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
        [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
        [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]
    ], dtype=np.float64)
    u, s, v = np.linalg.svd(A)
    X = v[:][8] / v[8][8]  # right singular vector
    H = np.reshape(X, (3, 3))  # make H a matrix
    H = H / H[2][2]  # normalize
    return H


def TwoDtohomogeneous(p_2D):
    """change the coordinates from 2D to 2D homogeneous"""
    assert p_2D.shape[1] == 2, "Input points stack has # of shape " + str(p_2D.shape)
    p_2D_homogeneous = np.ones([p_2D.shape[0], 3])
    p_2D_homogeneous[:, 0:2] = p_2D[:, :]
    p_2D_homogeneous = p_2D_homogeneous.astype(int)
    assert p_2D_homogeneous.shape[1] == 3
    return p_2D_homogeneous


def homogenousToTwoD(p_2D_homogeneous):
    """change the coordinates from 2D homogeneous to 2D"""
    assert p_2D_homogeneous.shape[1] == 3, "Input points stack has # of shape " + str(p_2D_homogeneous.shape)
    z = np.asarray(p_2D_homogeneous[:, 2]).reshape(p_2D_homogeneous.shape[0], 1)
    p_2D_homogeneous = p_2D_homogeneous / z  # make sure it is homogeneous
    p_2D = p_2D_homogeneous[:, 0:2]
    p_2D = p_2D.astype(int)
    assert p_2D.shape[1] == 2
    return p_2D


def perspectiveTransfer_image(img_perspect1, homo_from2to1, img_persoect2_or_size):
    """
    Change the image perspective from 1 to 2, replace the pixel value by the transformed view 1 image,
    if img form 2 is length 2 tuple, make a image that has row and col as first and second element of the tuple
    """
    img_perspect1 = np.asarray(img_perspect1)
    img_perspect1 = img_perspect1.astype(np.uint8)

    range_row = img_perspect1.shape[0]
    range_col = img_perspect1.shape[1]

    range_channel = 0
    if len(img_perspect1.shape) > 2:
        range_channel = img_perspect1.shape[2]
    """if input image is a integer, make one"""
    if type(img_persoect2_or_size) == tuple and len(img_persoect2_or_size) == 2:  # if input is not a image
        if range_channel == 0:
            img_perspect2 = np.zeros(img_persoect2_or_size)
        else:
            img_perspect2 = np.zeros([img_persoect2_or_size, img_persoect2_or_size, range_channel])
    else:  # if input is a image
        img_perspect2 = img_persoect2_or_size
        img_persoect2_or_size = img_persoect2_or_size.shape

    # img_perspect2 = cv2.warpPerspective(img_perspect1, homo, img_persoect2_or_size)
    homo_inv = np.linalg.inv(homo_from2to1)
    homo_inv = homo_inv / homo_inv[2, 2]
    for row in range(0, img_persoect2_or_size[0]):  # row
        for col in range(0, img_persoect2_or_size[1]):  # col
            pixel_img2 = np.array([[row, col, 1]]).T
            pixel_img1 = np.matmul(homo_inv, pixel_img2)  # from frame pixel to flatten view pixel
            pixel_img1 = ((pixel_img1 / pixel_img1[2])).astype(int)  # make it homogeneous coordinates
            # print(pixel_trans)
            if (pixel_img1[0] < range_row) and (pixel_img1[1] < range_col) and (pixel_img1[0] > -1) and (
                    pixel_img1[1] > -1):  # if the flatten view in bound
                img_perspect2[col, row] = img_perspect1[pixel_img1[1], pixel_img1[0]]
    img_perspect2 = img_perspect2.astype(np.uint8)
    return img_perspect2


def superimpose(img_dst, img_res, homo):
    (dst_rows, dst_cols, dst_channels) = img_dst.shape  # size of image that Lena being posted
    (res_rows, res_cols, res_channels) = img_res.shape  # size of Lena
    # img_dst = np.zeros([dst_rows, dst_cols, dst_channels])
    # img_dst_with_res = cv2.warpPerspective(img_res, homo, (res_rows, cols))     # get a black background image with Lena imposed
    # img_dst_with_res = np.transpose(img_dst_with_res, (1, 0, 2))
    """combine the image Lena being posted and the image with Lena imposed"""
    for row in range(0, res_rows):  # for each of the pixel on resource image
        for col in range(0, res_cols):
            pixel_img_res = np.array([row, col, 1]).T
            pixel_img_dst = np.dot(homo, pixel_img_res)  # from frame pixel to flatten view pixel
            pixel_img_dst = ((pixel_img_dst / pixel_img_dst[2])).astype(int)  # make it homogeneous coordinates
            # print("From " + str((row, col)) + " to " + str(pixel_img_dst), end=" ")
            # if (pixel_img_dst[0] > -1) \
            #         and (pixel_img_dst[1] > -1) \
            #         and (pixel_img_dst[0] <= dst_rows) \
            #         and (pixel_img_dst[1] <= dst_cols):   # if img_dst_with_res is black
            img_dst[pixel_img_dst[1], pixel_img_dst[0]] = img_res[col, row]  # copy the pixel value
            # print(" successful")
            # else:
            # print(" failure")
    return img_dst


def perspectiveTransfer_coord(p_2D, homo):
    """change the coordinates from 2D to 2D homogeneous
    the input points should be like:
    np.array([[x0, y0],
    [x1, y1],
    ...
    ])
    output points follow the same format
    """
    assert p_2D.shape[1] == 2, "Input points stack has shape: " + str(p_2D.shape)
    p_2D_homo = TwoDtohomogeneous(p_2D)
    assert p_2D_homo.shape[1] == 3, "After homogeneous, input points stack has shape: " + str(p_2D_homo.shape)
    p_2D_homo_transfromed = np.transpose(
        np.dot(homo, p_2D_homo.transpose()))  # make first step of homograph transformation
    p_2D_transfromed = homogenousToTwoD(p_2D_homo_transfromed)
    """
    z = np.asarray(p_2D_homo_transfromed[:, 2]).reshape(p_2D_homo_transfromed.shape[0], 1)  # third element after first homograph transformation
    p_2D_homo_transfromed = p_2D_homo_transfromed/z    # divide third element, third element will be 1
    p_2D_transfromed = p_2D_homo_transfromed[:, 0:2]
    p_2D_transfromed = p_2D_transfromed.astype(int)     # force pixel coordinates to be integer
    """

    assert p_2D_transfromed.shape[1] == 2, "Output points stack has # of shape " + str(p_2D_transfromed.shape)
    return p_2D_transfromed


def shiftToASpot(points, spot):
    """
    get a set of points in rows: [[x1, y1][x2, y2][x3, y3][x4, y4]...],
    and change the frame of coordinates to the geometrical center of points
    by subtract each points coordinates by the mean
    """
    assert points.shape[1] == spot.shape[0], "You wanna move from dimension of " + str(
        points.shape) + " to dimension of " + str(spot.shape)
    center = np.mean(points, axis=0)
    points_shift = points - center + spot
    assert center.shape[0] == points.shape[1]  # make sure the mean were taken along column
    assert points_shift.shape == points.shape
    return points_shift, center


def projectionMatrix(intrinsic_camera_matrix, homo):
    Kinv = np.linalg.inv(intrinsic_camera_matrix)
    B = np.dot(Kinv, homo)

    det_B = np.linalg.det(B)
    if det_B < 0:
        B = B * -1
    elif det_B > 0:
        B = B
    b1 = B[:, 0]
    b2 = B[:, 1]
    b3 = B[:, 2]
    Lamda = 2 / (np.linalg.norm(b1) + np.linalg.norm(b2))
    r1 = b1 * Lamda
    r2 = b2 * Lamda
    r3 = np.cross(r1, r2)
    t = (b3 * Lamda).reshape([3, 1])
    r = np.array([r1, r2, r3]).transpose()
    homogeneousTransformationMatrix = np.hstack([r, t])
    perjectionMatrix = np.dot(intrinsic_camera_matrix, homogeneousTransformationMatrix)
    return perjectionMatrix / perjectionMatrix[2, 3]


def invHomo(homo):
    assert homo.shape == (3, 3)
    homo_inv = np.linalg.inv(homo)
    homo_inv = homo_inv / homo_inv[2, 2]
    return homo_inv
