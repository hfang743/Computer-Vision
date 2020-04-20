import numpy as np
from __main__ import *

def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def cameraParamsPt2():
    # Camera Matrix
    K=np.array([
        [9.037596e+02,0.000000e+00,6.957519e+02],
        [0.000000e+00,9.019653e+02, 2.242509e+02],
        [0.000000e+00, 0.000000e+00,1.000000e+00]
    ])
    # distortion coefficients
    D=np.array([-3.639558e-01,1.788651e-01,6.029694e-04, -3.922424e-04, - 5.382460e-02])

    return K,D

def cameraParamsPt3():
    # Camera Matrix
    K =np.array([
        [1.15422732e+03,   0.00000000e+00,   6.71627794e+02],
        [0.00000000e+00,   1.14818221e+03,   3.86046312e+02],
        [0.00000000e+00,  0.00000000e+00,   1.00000000e+00]
    ])

    # Distortion Coefficients
    D = np.array([-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])

    return K,D

def adjustSaturation(frame,hue,sat):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv=hue*hsv+sat
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return frame

def adjustContrast(frame, contrast):

        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        frame = cv2.addWeighted(frame, alpha_c, frame, 0, gamma_c)

        return frame

def yellowAndWhite(frame):
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    whitemask = cv2.inRange(grayframe, 200, 255)
    grayframe = cv2.bitwise_or(grayframe, grayframe, mask=whitemask)

    # yellow mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_val = (21, 39, 64)
    high_val = (40, 255, 255)
    mask = cv2.inRange(hsv, low_val, high_val)
    frame = cv2.bitwise_or(hsv, hsv, mask=mask)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    yellowframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = yellowframe + grayframe
    whitemask = cv2.inRange(frame, 150, 255)
    frame = cv2.bitwise_or(frame, frame, mask=whitemask)

    return frame