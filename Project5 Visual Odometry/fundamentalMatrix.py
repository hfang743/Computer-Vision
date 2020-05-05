from __main__ import *
import numpy as np
import cv2
# from test2 import *
from sklearn.preprocessing import normalize


def Normalize(points, N):
    avg = np.average(points, axis=0)
    mean = points - avg.reshape(1, 2)
    sum = np.sum((mean) ** 2, axis=None)
    std = 1 / ((sum / (2 * N)) ** 0.5)
    NormP = mean * std
    return NormP, std, avg


def DeNormalize(std, mean):
    X = np.zeros((3, 3))
    X[0, 0] = std
    X[1, 1] = std
    X[2, 2] = 1
    X[0, 2] = -std * mean[0]
    X[1, 2] = -std * mean[1]
    return X


def getrand8(p1, p2):
    indexLin = np.arange(0, len(p1), 1)
    np.random.shuffle(indexLin)

    Indx8 = indexLin[0:8]

    p1 = p1[Indx8, :]
    p2 = p2[Indx8, :]

    return p1, p2


def enforceRank(var):
    u, s, v = np.linalg.svd(var)
    s[-1] = 0
    F = np.dot(u, np.dot(np.diag(s), v)).T
    # if F[2,2]!=1:
    #     print('WARNING BAD F: \n',F)

    return F


def createFund(p1, p2):
    N = p1.shape[0]

    # NORMALIZE
    x, xstd, xavg = Normalize(p1, N)
    y, ystd, yavg = Normalize(p2, N)

    A = np.ones((N, 9))
    A[:, 0:2] = x * y[:, 0].reshape(N, 1)
    A[:, 2] = y[:, 0]
    A[:, 3:5] = x * y[:, 1].reshape(N, 1)
    A[:, 5] = y[:, 1]
    A[:, 6:8] = x

    u, s, vt = np.linalg.svd(A, full_matrices=True)
    F = vt[8, :].reshape(3, 3)

    # ENFORCE RANK
    F = enforceRank(F)

    # DENORMALIZE
    X1 = DeNormalize(xstd, xavg)
    X2 = DeNormalize(ystd, yavg)

    F = np.dot(X2.T, np.dot(F, X1))

    return F / F[2, 2]


def computeFMatrix(p1, p2):
    # P1=Prev
    # P2=Cur

    eta = 1
    SinTotal = 0
    #
    for i in range(200):

        # if i % 100 == 0:
        #     print(i)

        p18, p28 = getrand8(p1, p2)

        H1 = np.ones([len(p1), 1])
        p1H = np.append(p1, H1, axis=1)
        p2H = np.append(p2, H1, axis=1)

        FC = createFund(p18, p28)

        Sin = 0
        mask = []
        for j in range(len(p1)):
            X1 = p1H[j]
            X2 = p2H[j]
            # Error = np.matmul(np.matmul(X2.T, FC), X1)
            Error = np.matmul(X2.T, np.matmul(FC, X1))
            cV = np.abs(Error)
            if cV < eta:
                Sin += 1
                mask.append(True)
            else:
                mask.append(False)

        if Sin > SinTotal:
            SinTotal = Sin
            maskIn = mask

            F = FC

            p1o = p1[maskIn]
            p2o = p2[maskIn]

    F = createFund(p1o, p2o)
    # F = enforceRank(F)
    # print('#######################################################')
    # print('Inliers: ', SinTotal)
    # print('p1 shape: ', p1.shape[0])
    # print('% Inliers: ', 100 * SinTotal / p1.shape[0])
    #
    # print('\nCustom F\n', F)
    # FCV, CV2mask = cv2.findFundamentalMat(p1, p2, cv2.RANSAC)
    #
    # print('\nCV2 F\n', FCV)
    # FMat, _, _ = ransac_fundamental_matrix(p1, p2)
    # print('\nMat F\n', FMat)
    #
    # # print('\nF Diff: \n', np.abs(FCV - F))
    #
    # print('\nCv2 Mask:\n', CV2mask[:100].T)
    # print('\nCust Mask:\n', mask[:100])n

    F = enforceRank(F)
    hold = 1

    return F, p1o, p2o
