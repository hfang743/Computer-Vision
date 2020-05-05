import cv2
import os
import numpy as np
# for filename in os.listdir('./Oxford_dataset/stereo/centre/'):
#     print(filename)

img1 = cv2.imread('./Oxford_dataset/stereo/centre/1399381446392174.png')
img2 = cv2.imread('./Oxford_dataset/stereo/centre/1399381446454665.png')


def MatchFeatures(img1, img2):
    # orb = cv2.ORB_create(nfeatures=1000)
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # # Brute force Matcher
    # # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    #
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    # cv2.imshow('Matches', match_img)
    # cv2.waitKey()
    ############################################
    # FLANN matcher
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None)

    pointsmatched = []
    pointsfrom1 = []
    pointsfrom2 = []

    for i, (s, p) in enumerate(matches):
        if s.distance < 1 * p.distance:
            pointsmatched.append(s)
            pointsfrom2.append(kp2[s.trainIdx].pt)
            pointsfrom1.append(kp1[s.queryIdx].pt)

    p1A = np.int32(pointsfrom1)
    p2A = np.int32(pointsfrom2)



    # Select points to match
    # features = []
    # for m, n in matches:
    #     if m.distance < 0.5 * n.distance:
    #         features.append([m])
    # match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, features, None, flags=2)

    # imgs = np.vstack([knn_image, match_img])
    # cv2.namedWindow("Matches",0);
    # cv2.resizeWindow("Matches", 1000, 600);
    # cv2.imshow('Matches', imgs)
    # cv2.waitKey(0)

    return p1A, p2A


def matchUntil(img1, img2, sift):
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None)

    pointsmatched = []
    pointsfrom1 = []
    pointsfrom2 = []

    for i, (s, p) in enumerate(matches):
        if s.distance < .5 * p.distance:
            pointsmatched.append(s)
            pointsfrom2.append(kp2[s.trainIdx].pt)
            pointsfrom1.append(kp1[s.queryIdx].pt)

    pointsfrom1 = np.float32(pointsfrom1)
    pointsfrom2 = np.float32(pointsfrom2)

    # Select points to match
    # features = []
    # for m, n in matches:
    #     if m.distance < 0.5 * n.distance:
    #         features.append([m])
    # match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, features, None, flags=2)
    #
    # imgs = np.vstack([knn_image, match_img])
    # cv2.namedWindow("Matches",0);
    # cv2.resizeWindow("Matches", 1000, 600);
    # cv2.imshow('Matches', imgs)
    # cv2.waitKey(0)

    return pointsfrom1, pointsfrom2
