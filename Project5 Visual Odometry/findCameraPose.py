"""
Created on Thu April 25 13:00:08 2020
@author: Cheng Chen
"""
import numpy as np

flag_DisambiguateCameraPose = True


def homoTransformation(rotation, translation, homo=True):
    homoTransf = np.dot(rotation, np.hstack((np.eye(3), -translation)))
    # homoTransf = np.dot((rotation, -np.dot(rotation, translation)), axis=1)
    if homo:
        homoTransf = np.concatenate((homoTransf, np.array([[0, 0, 0, 1]])), axis=0)
    return homoTransf


def essentialMatrixFromFundamentalMatrix(fundamentalMatrix, cameraIntrinsicMatrix):
    """
    This function compute e essential matrix from F and K.
    As in the case of F matrix computation, the singular values of E are not necessarily
    (1,1,0) due to the noise in K. This can be corrected by reconstructing it with (1,1,0)
    singular values.
    It is important to note that the F is defined in the original image space (i.e. pixel coordinates) whereas E is in
    the normalized image coordinates. Normalized image coordinates have the origin at the optical center of the image.
    Also, relative camera poses between two views can be computed using E matrix. Moreover, F has 7 degrees of freedom
    while E has 5 as it takes camera parameters in account.

    :param fundamentalMatrix: matrix is only an algebraic representation of epipolar geometry and can both geometrically (contructing the epipolar line) and arithematically
    :param cameraIntrinsicMatrix: camera calibration matrix or camera intrinsic matrix
    :return: essentialMatrix: Essential matrix is another 3×3 matrix, but with some additional properties that relates
    the corresponding points assuming that the cameras obeys the pinhole model (unlike fundamental matrix)
    """
    assert type(fundamentalMatrix) == np.ndarray and type(cameraIntrinsicMatrix) == np.ndarray
    assert fundamentalMatrix.shape == (3, 3) and cameraIntrinsicMatrix.shape == (3, 3)
    assert np.linalg.matrix_rank(fundamentalMatrix) == 2, "The rank of Fundamental Matrix should be 2, but got " + str(
        np.linalg.matrix_rank(fundamentalMatrix)) + "instead"
    Sigma_new = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 0]])  # use to reconstructing esstential matrix singular values to be (1,1,0)
    """compute the crude esstntial matrix"""
    essentialMatrix = cameraIntrinsicMatrix.T @ fundamentalMatrix @ cameraIntrinsicMatrix
    """correct the essential matrix using signal value decompose"""
    U, Sigma, V_transposed = np.linalg.svd(essentialMatrix)
    Sigma = np.diag(Sigma)
    if np.linalg.norm(essentialMatrix - (U @ Sigma @ V_transposed)) > 1:  # check svd works as it supposed to be
        print("SVD of Essential Matrix doesn't match up")
        print("Essential Matrix original is \n" + str(essentialMatrix))
        print("U is\n" + str(U) + "\nSigma is\n" + str(Sigma) + "\nV_transposed is\n" + str(V_transposed))
        print("SVD of Essential Matrix product \n" + str(U @ Sigma @ V_transposed))
        # raise ValueError("Essential Matrix and SVD of Essential Matrix product doesn't match")
        print("Essential Matrix and SVD of Essential Matrix product doesn't match")
    essentialMatrix = U @ Sigma_new @ V_transposed
    return essentialMatrix


def extractCameraPoseFromEssntialMatrix(essentialMatrix):
    """
    :param essentialMatrix: Essential matrix is another 3×3 matrix, but with some additional properties that relates
    the corresponding points assuming that the cameras obeys the pinhole model (unlike fundamental matrix)
    :return:
    """
    assert type(essentialMatrix) == np.ndarray
    assert essentialMatrix.shape == (3, 3)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  #
    U, Sigma, V_transposed = np.linalg.svd(essentialMatrix)
    """The decomposition is not unique. We will assume that we have a singular value decomposition where det(U V^T) =
    1. It is easy to ensure this; If we have an SVD as in (15) with det(U V^T) = −1 then we can simply switch the sign 
    of the last column of V . Alternatively we can switch to −E which then has the SVD
    """
    # print("Determinant is " + str(int(np.linalg.det(U @ V_transposed))))
    # if int(np.linalg.det(U @ V_transposed)) == -1:
    #     # V_transposed[-1, :] = - V_transposed[-1, :]
    #     U, Sigma, V_transposed = np.linalg.svd(-essentialMatrix)
    #     print("Determinant is " + str(int(np.linalg.det(U @ V_transposed))))

    """There are four camera pose configurations, (C1, R1), (C2, R2), (C3, R3), (C4, R4)"""
    C1 = U[:, 2].reshape(3, 1)
    R1 = U @ W @ V_transposed
    R3 = U @ W.T @ V_transposed
    """choose one camera pose and rotation matrix from four"""
    rotationMatrixes = [R1, R1, R3, R3]
    camerPosistions = [C1, -C1, C1, -C1]
    """det(R)=1 . If det(R)=−1, the camera pose must be corrected i.e. C=−C and R=−R."""
    for i in range(4):
        if np.linalg.det(rotationMatrixes[i]) < 0:
            rotationMatrixes[i] = -rotationMatrixes[i]
            camerPosistions[i] = -camerPosistions[i]
    assert R1.shape == (3, 3) and C1.shape == (3, 1)
    return rotationMatrixes, camerPosistions


def skew(point):
    assert point.shape == (3, 1)
    point = np.squeeze(point.T)
    return np.array([[0, -point[2], point[1]], [point[2], 0, -point[0]], [-point[1], point[0], 0]])


def linearTriangulation(point_1, projectionMatrix_1, point_2, projectionMatrix_2):
    """
    Triangulate the 3D points, given two camera poses and correpsounding projection matrix.
    :param point_1: np,ndarray <int> first point of pair in camera coordinates
    :param projectionMatrix_1: np,ndarray <float> projection matrix that map the first 2D point to 3D
    :param point_2: np,ndarray <int> second point of pair in camera coordinates
    :param projectionMatrix_2: np,ndarray <float> projection matrix that map the first 2D point to 3D
    :return: the 3D point that this pair corresponding to
    """
    assert type(point_1) == type(point_2) == type(projectionMatrix_1) == type(projectionMatrix_2) == np.ndarray
    assert point_1.shape == point_2.shape == (3,) or point_1.shape == point_2.shape == (3, 1), "point_1 shape: " + str(
        point_1.shape) + " point_2 shape: " + str(point_2.shape)
    assert projectionMatrix_1.shape == projectionMatrix_2.shape == (3, 4)
    point_1 = point_1.reshape((3, 1))
    point_2 = point_2.reshape((3, 1))
    """skew matrix method"""
    # NumCoorespoundences = point_1.shape[0]  # how many correspondences
    point_1_skew = skew(point_1)
    point_2_skew = skew(point_2)
    D_matrix = np.vstack((np.dot(point_1_skew, projectionMatrix_1), np.dot(point_2_skew, projectionMatrix_2)))
    """slove for the 3D point coordinate"""
    _, _, V_transposed = np.linalg.svd(D_matrix)
    point_3D = V_transposed[-1, :] / V_transposed[
        -1, -1]  # last row of V transposed correspounding to the smallest singular value
    point_3D = point_3D.reshape(4, -1)
    assert point_3D[3, 0] == 1
    return point_3D


def linearTriangulation_1(point_1, projectionMatrix_1, point_2, projectionMatrix_2):
    """
    Triangulate the 3D points, given two camera poses and correpsounding projection matrix.
    :param point_1: np,ndarray <int> first point of pair in camera coordinates
    :param projectionMatrix_1: np,ndarray <float> projection matrix that map the first 2D point to 3D
    :param point_2: np,ndarray <int> second point of pair in camera coordinates
    :param projectionMatrix_2: np,ndarray <float> projection matrix that map the first 2D point to 3D
    :return: the 3D point that this pair corresponding to
    """
    assert type(point_1) == type(point_2) == type(projectionMatrix_1) == type(projectionMatrix_2) == np.ndarray
    assert point_1.shape == point_2.shape == (3,) or point_1.shape == point_2.shape == (3, 1), "point_1 shape: " + str(
        point_1.shape) + " point_2 shape: " + str(point_2.shape)
    assert projectionMatrix_1.shape == projectionMatrix_2.shape == (3, 4)
    point_1 = point_1.reshape((3, 1))
    point_2 = point_2.reshape((3, 1))
    """matrix multiplication method"""
    D_matrix_1 = (point_1[0, :] * projectionMatrix_1[2, :] - projectionMatrix_1[0, :]).reshape(1, 4)
    D_matrix_2 = (point_1[1, :] * projectionMatrix_1[2, :] - projectionMatrix_1[1, :]).reshape(1, 4)
    D_matrix_3 = (point_2[0, :] * projectionMatrix_2[2, :] - projectionMatrix_2[0, :]).reshape(1, 4)
    D_matrix_4 = (point_2[1, :] * projectionMatrix_2[2, :] - projectionMatrix_2[1, :]).reshape(1, 4)
    D_matrix = np.concatenate((D_matrix_1, D_matrix_2, D_matrix_3, D_matrix_4), axis=0)
    assert D_matrix.shape == (4, 4), "D matrix shape: " + str(D_matrix.shape)
    """slove for the 3D point coordinate"""
    _, _, V_transposed = np.linalg.svd(D_matrix)
    point_3D = V_transposed[-1, :] / V_transposed[
        -1, -1]  # last row of V transposed correspounding to the smallest singular value
    point_3D = point_3D.reshape(len(point_3D), -1)
    assert point_3D[3, 0] == 1
    return point_3D


def DisambiguateCameraPose(rotations_fromLastToCurrentFrame, translations_fromLastToCurrent, cameraIntrinsicMatrix,
                           points_currentFrame, points_lastFrame):
    """
    Given a camera pose configurations and their the 2D points correspondences, find how many points pass the cheirality condition
    :param homoTransf: the homogeneous transformation between last frame and current frame
    :param cameraIntrinsicMatrix:
    :param points_currentFrame: last frame
    :param points_lastFrame: current frame
    :return: number of correspondences pass the cheirality condition
    """
    assert type(points_currentFrame) == type(points_lastFrame) == type(cameraIntrinsicMatrix) == type(
        rotations_fromLastToCurrentFrame[0]) == type(translations_fromLastToCurrent[0]) == np.ndarray
    assert type(points_currentFrame[0, 0]) == type(points_lastFrame[0, 0]) == np.float32
    assert points_currentFrame.shape[1] == points_lastFrame.shape[1] == 2
    assert cameraIntrinsicMatrix.shape == (3, 3)
    assert rotations_fromLastToCurrentFrame[0].shape == (3, 3)
    assert translations_fromLastToCurrent[0].shape == (3, 1)
    """format data make points set from (n, 2) to (n, 3)"""
    points_currentFrame = np.transpose(np.insert(points_currentFrame, 2, 1, axis=1))
    points_lastFrame = np.transpose(np.insert(points_lastFrame, 2, 1, axis=1))

    """get homogeneous, projection transformation between last frame and last frame, and current frame ready"""
    homoTransf_toSelf = np.concatenate((np.eye(3), np.array([[0], [0], [0]])), axis=1)
    projectionMatrix_toSelf = cameraIntrinsicMatrix @ homoTransf_toSelf
    votes = []
    for i in range(len(rotations_fromLastToCurrentFrame)):
        """for each combination of rotation and translation"""
        homoTransf_fromLastToCurrentFrame = homoTransformation(rotations_fromLastToCurrentFrame[i],
                                                               translations_fromLastToCurrent[i],
                                                               homo=False)  # rigid transformation
        projectionMatrix_fromLastToCurrentFrame = cameraIntrinsicMatrix @ homoTransf_fromLastToCurrentFrame  # form a project matrix from second frame

        """loop over all points, vote for the points pass cheirality condition check"""
        vote = 0
        for point_currentFrame, point_lastFrame in zip(points_currentFrame.T, points_lastFrame.T):
            """reconstruct 3D points in camera coordinates of last frame"""
            point_3D_inPreviousView = linearTriangulation(point_currentFrame, projectionMatrix_toSelf, point_lastFrame,
                                                          projectionMatrix_fromLastToCurrentFrame)  # shape (4, 1)
            """reconstruct 3D points in camera coordinates of current frame"""
            z = np.dot(rotations_fromLastToCurrentFrame[i][2, 0:3],
                       point_3D_inPreviousView[0:3] - translations_fromLastToCurrent[i])
            if z > 0 and point_3D_inPreviousView[2, 0] > 0 and translations_fromLastToCurrent[i][
                2] >= 0:  # the depth (z compound of 3D point)
                vote = vote + 1
        votes.append(vote)
    """inbuilt function to find the position of maximum vote"""
    correct_index = votes.index(max(votes))
    return rotations_fromLastToCurrentFrame[correct_index], translations_fromLastToCurrent[
        correct_index], votes, correct_index

# def findSecondFrameCameraPoseFromFirstFrame(fundamentalMatrix, cameraIntrinsicMatrix, points_currentFrame, points_lastFrame):
#     """
#     Based on the refined fundamental matrix, camera intrinsic matrix, and some points correspondences, find the camera
#     pose of first frame from the second frame
#     :param fundamentalMatrix: matrix is only an algebraic representation of epipolar geometry and can both geometrically (contructing the epipolar line) and arithematically
#     :param cameraIntrinsicMatrix: camera calibration matrix or camera intrinsic matrix
#     :param points_currentFrame: a stack of coordinates from first frame, which is the 3D points map to 2D points, and
#                             sequentially corresponding to coordinates in second frame
#     :param points_lastFrame: a stack of coordinates from second frame, which is the 3D points map to 2D points, and
#                             sequentially corresponding to coordinates in first frame
#     :return: the rotation and translation matrix between camera where it takes first and where it takes second frame
#     """
#     assert type(fundamentalMatrix) == type(cameraIntrinsicMatrix) == type(points_currentFrame) == type(points_lastFrame) == np.ndarray
#     assert points_currentFrame.shape[1] == points_currentFrame.shape[1] == 2
#     assert fundamentalMatrix.shape == cameraIntrinsicMatrix.shape == (3, 3)
#
#     """get essential matrix"""
#     essentialMatrix = essentialMatrixFromFundamentalMatrix(fundamentalMatrix, cameraIntrinsicMatrix)
#     """get four possible rotation matrices and translation matrices"""
#     rotations_fromLast, cameraTranslations_fromLast = extractCameraPoseFromEssntialMatrix(essentialMatrix)
#     """pick one of the four possible rotation matrices and translation matrices, using cheirality condition check"""
#     rotation_fromLast, cameraTranslations_fromLast = DisambiguateCameraPose(rotations_fromLast, cameraTranslations_fromLast, cameraIntrinsicMatrix, points_currentFrame, points_lastFrame)
#
#     return rotation_fromLast, cameraTranslations_fromLast
