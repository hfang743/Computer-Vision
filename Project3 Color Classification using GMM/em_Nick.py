import numpy as np
import cv2
from scipy.stats import multivariate_normal as mvn
import random
from imutils import contours
import pickle


# compute the pdf of a n-Dimension gaussian distribution
def GaussianPDF(x, mean, covar):
    dimension = x.size
    x = x.reshape((1, dimension))
    mean = mean.reshape((1, dimension))
    offset = x - mean  # offset between the point and mean
    pdf = np.exp(-1 / 2 * np.dot(np.dot(offset, np.linalg.inv(covar)), offset.T))
    pdf = 1 / ((2 * np.pi) ** (dimension / 2) * np.sqrt(np.linalg.det(covar))) * pdf
    pdf = float(pdf[0, 0])

    return pdf


# compute the conditional probability of a color label given the color observation
def responsibilities(point_set, mean_set, covar_set, alpha_set, dimension, k):
    point_list = point_set
    mean_list = mean_set
    covar_list = covar_set

    # extract parameters
    k = len(mean_set)
    # compute the responsibilities before normalization
    respons = []  # (k, N) list
    for index_cluster in range(k):
        mean = mean_set[index_cluster]
        covar = covar_set[index_cluster]
        alpha = alpha_set[index_cluster]
        try:
            weightedLikelihoods = [alpha * GaussianPDF(point, mean, covar) for point in point_set]
        except:
            mean_list, covar_list, alpha_list = initilizeGassuainClusterModelParameters(point_list, dimension, k)
            print("Rebuilding: Failure in cluster Initilization - Singular Matrix")

            point_list, mean_list, covar_list, respons = responsibilities(point_list, mean_list, covar_list, alpha_list,
                                                                          dimension, k)
        respons.append(weightedLikelihoods)
    # responsibilities normalization
    sums_responsibilityAllCluster = [sum(responsibility) for responsibility in
                                     zip(*respons)]  # for each of data point, get sum of all culster responsibility
    for index_cluster in range(k):
        """for each responsibility, divide it by the sum of responsibility from all culsters"""
        respon = respons[index_cluster]

        try:
            respons[index_cluster] = [(responsibility / sums_responsibility_eachPooint_AllCluster) for
                                      responsibility, sums_responsibility_eachPooint_AllCluster in
                                      zip(respon, sums_responsibilityAllCluster)]
            # print("Found good cluster")
            # shape check
            assert len(respons) == len(mean_set)  # check k
            assert type(respons[0]) == list  # check type of list
            assert type(respons[0][0]) == float  # check responsibility type
            assert len(respons[0]) == len(point_set)  # check number of sample
        except:
            mean_list, covar_list, alpha_list = initilizeGassuainClusterModelParameters(point_list, dimension, k)
            print("Rebuilding: Failure in cluster Initilization - Divide by zero")

            point_list, mean_list, covar_list, respons = responsibilities(point_list, mean_list, covar_list, alpha_list,
                                                                          dimension, k)

    return point_list, mean_list, covar_list, respons


# initialization of means and covariances
def initilizeGassuainClusterModelParameters(point_list, dimension, k):
    alpha_list = [1 / k] * k
    mean_list = random.choices(point_list, k=k)
    covar_list = [np.eye(dimension, dimension) for index_cluster in range(k)]
    for index, covar in enumerate(covar_list):
        var_rand = 255 * np.random.rand(dimension)
        var_rand = list(var_rand.astype(int))
        np.fill_diagonal(covar, var_rand)
        covar_list[index] = covar

    return mean_list, covar_list, alpha_list


# find the parameters weight factors, means, covariances such that those would maximize the corectness of  conditional probability of a color label given the color observation.
def likelyhoodMaximization(point_list, k):
    NumSample = len(point_list)
    dimension = point_list[0].size
    # initialization of means and covariances
    mean_list, covar_list, alpha_list = initilizeGassuainClusterModelParameters(point_list, dimension, k)
    for interator in range(1000):
        # calculate the normalized responsibilities of each point of each cluster: first index cluster, second index point
        point_list, mean_list, covar_list, respons = responsibilities(point_list, mean_list, covar_list, alpha_list,
                                                                      dimension, k)
        # update means, covariances, and weight factors
        mean_previous_list = mean_list.copy()  # store the previous means
        point_array = np.asarray(point_list)  # cast list of points to array of points

        for index in range(k):  # for each cluster
            respon = respons[index]
            """update means"""
            respon_array = np.asarray(respon).reshape((NumSample, 1))
            mean_list[index] = (np.dot(point_array.T, respon_array) / sum(respon)).reshape((dimension))
            """update covariance"""
            # compute the covariance of each point from point list
            covar_list_eachPoint = [np.dot((point.reshape((1, dimension)) - mean_list[index].reshape((1, dimension))).T,
                                           (point.reshape((1, dimension)) - mean_list[index].reshape((1, dimension)))) for point in
                                    point_list]
            # sum up all the product of each point's responsibility and covariance
            covar_sum = sum([respon_onePoint * covar_onePoint for respon_onePoint, covar_onePoint in
                             zip(respon, covar_list_eachPoint)])
            covar_list[index] = covar_sum / sum(respon)
            """update weight factor"""
            alpha_list[index] = 1 / NumSample * sum(respon)
        """check the stop criteria"""
        Convergence = sum(
            [np.linalg.norm(mean - mean_previous) for mean, mean_previous in zip(mean_list, mean_previous_list)])
        if Convergence > 0.01:
            continue
        else:
            print("Convergenced")
            break

    return mean_list, covar_list, alpha_list


def genLogLike(image, mean, Sigma, w, nx, ny, div, K):
    likelihoods = np.zeros((K, nx * ny))
    log_likelihood = np.zeros(nx * ny)
    for k in range(K):
        likelihoods[k] = w[k] * mvn.pdf(image, mean[k], Sigma[k], allow_singular=True)
        # ((2.0 * np.pi) ** (-d / 2.0)) * (1.0 / (np.linalg.det(Sigma[k]) ** 0.5)) * np.exp(-0.5 * np.sum(np.multiply(x_mean * Sinv, x_mean), axis=1))
        log_likelihood = likelihoods.sum(0)
        # likelihoods[:,k:k+1] = w[k] *GaussianPDF(image, mean[k], Sigma[k])
        # log_likelihood = likelihoods.sum(1)
    log_likelihood = np.reshape(log_likelihood, (nx, ny))
    log_likelihood[log_likelihood > np.max(log_likelihood) / div] = 255

    return log_likelihood


def getLikelihood(vid, K, color):
    point_list = np.load(vid)

    mean, Sigma, w = likelyhoodMaximization(list(point_list), K)
    w = np.asarray(w)
    Sigma = np.asarray(Sigma)
    mean = np.asarray(mean)
    return mean, Sigma, w, K

def saveVar(var,varprint):
    with open(varprint, 'wb') as f:
        pickle.dump(var, f)

def openVar(varprint):
    with open(varprint, 'rb') as f:
        var= pickle.load(f)
    return var



def em_NickMain(LearnFlag):
    if LearnFlag:
        print("Getting Orange Parameters")
        meanO, SigmaO, wO, KO = getLikelihood('OTrain.npy', 3, color='o')

        print("Getting Yellow Parameters")
        meanY,SigmaY,wY, KY = getLikelihood('YTrain.npy', 2, color='y')

        print("Getting Green Parameters")
        meanG,SigmaG,wG, KG = getLikelihood('GTrain.npy', 2, color='g')

        print("All parameters attained")


        saveVar(meanO,'meanO')
        saveVar(meanY,'meanY')
        saveVar(meanG,'meanG')

        saveVar(SigmaO,'SigmaO')
        saveVar(SigmaY,'SigmaY')
        saveVar(SigmaG,'SigmaG')

        saveVar(wO,'wO')
        saveVar(wY,'wY')
        saveVar(wG,'wG')

        saveVar(KO,'KO')
        saveVar(KY,'KY')
        saveVar(KG,'KG')

        print('Done Saving off parameter')

    else:
        meanO=openVar('meanO')
        meanY=openVar('meanY')
        meanG=openVar('meanG')

        SigmaO=openVar('SigmaO')
        SigmaY=openVar('SigmaY')
        SigmaG=openVar('SigmaG')

        wO=openVar('wO')
        wY=openVar('wY')
        wG=openVar('wG')

        KO=openVar('KO')
        KY=openVar('KY')
        KG=openVar('KG')

    print("detecting")

    ######################################### Output #########################################

    name = "detectbuoy.avi"
    cap = cv2.VideoCapture(name)
    images = []
    while (cap.isOpened()):
        success, frame = cap.read()
        if success == False:
            break

        nx, ny, ch = frame.shape
        image = np.reshape(frame, (nx * ny, ch))

        log_likelihoodO = genLogLike(image, meanO, SigmaO, wO, nx, ny, 3, KO)
        log_likelihoodY = genLogLike(image, meanY, SigmaY, wY, nx, ny, 5, KY)
        log_likelihoodG = genLogLike(image, meanG, SigmaG, wG, nx, ny, 3, KG)

        outputO = np.zeros_like(frame)
        outputY = np.zeros_like(frame)
        outputG = np.zeros_like(frame)
        # output[:, :, 0] = log_likelihood
        # output[:, :, 1] = log_likelihood
        outputO[:, :, 2] = log_likelihoodO
        outputY[:, :, 2] = log_likelihoodY
        outputG[:, :, 1] = log_likelihoodG

        blurO = cv2.GaussianBlur(outputO, (3, 3), 40)
        blurO = cv2.GaussianBlur(blurO, (3, 3), 40)
        edgeO = cv2.Canny(blurO, 10, 255)
        cntsO, _ = cv2.findContours(edgeO, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts_sortedO, _ = contours.sort_contours(cntsO, method="left-to-right")
        hullO = cv2.convexHull(cnts_sortedO[0])
        (xO, yO), radO = cv2.minEnclosingCircle(hullO)
        xO = int(xO)
        yO = int(yO)
        # cv2.imshow("Orange", edgeO)

        blurY = cv2.GaussianBlur(outputY, (3, 3), 5)
        edgeY = cv2.Canny(blurY, 50, 255)
        cntsY, _ = cv2.findContours(edgeY, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts_sortedY, _ = contours.sort_contours(cntsY, method="left-to-right")
        hullY = cv2.convexHull(cnts_sortedY[0])
        (xY, yY), radY = cv2.minEnclosingCircle(hullY)
        xY = int(xY)
        yY = int(yY)
        # cv2.imshow("Yellow", outputY)

        blurG = cv2.GaussianBlur(outputG, (3, 3), 20)
        edgeG = cv2.Canny(blurG, 20, 255)
        cntsG, _ = cv2.findContours(edgeG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts_sortedG, _ = contours.sort_contours(cntsG, method="left-to-right")
        hullG = cv2.convexHull(cnts_sortedG[0])
        (xG, yG), radG = cv2.minEnclosingCircle(hullG)
        xG=int(xG)
        yG=int(yG)
        # cv2.imshow("Green", outputG)



        YO=np.real(np.sqrt((xO-xY)^2+(yO-yY)^2))
        print('YO',YO)
        YG=np.real(np.sqrt((xG-xY)^2+(yG-yY)^2))
        print('YG',YG)

        if radY > 7:
            cv2.circle(frame, (xY, yY), int(radY), (0, 255, 255), 4)


            images.append(frame)

        if radO > 5 and YO>6:
            cv2.circle(frame, (xO, yO), int(radO), (0, 0, 255), 4)


            images.append(frame)

        if radG > 5 and YG>6:
            cv2.circle(frame, (xG, yG), int(radG), (0, 255, 0), 4)


            images.append(frame)

        else:

            images.append(frame)

        cv2.imshow("Final output", frame)
        cv2.waitKey(1)

    cap.release()

    #Up to date
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('3D_gauss_orange.avi', fourcc, 5.0, (640, 480))
    # for image in images:
    #     out.write(image)
    #     cv2.waitKey(10)

    # out.release()
