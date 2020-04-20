import numpy as np
import cv2
from __main__ import *
import matplotlib.pyplot as plt
import imutils


def historgramOnYAxis(img_gray):
    """calculate the historgram of pixel value along y axis"""
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    assert len(img_gray.shape) == 2
    # hist = np.zeros(img_gray.shape[1])
    hist = np.mean(img_gray, axis=0)
    assert hist.shape[0] == img_gray.shape[1]
    return hist


def findLeftAndRightPoints(img_gray):
    hist = historgramOnYAxis(img_gray)

    ignoreRegion = 80  # how many pixel rows to be ignore around the highest value pixel

    index_first = np.argmax(hist)
    hist_noFirstPeak = hist.copy()
    hist_noFirstPeak[index_first - ignoreRegion: index_first + ignoreRegion] = 0
    index_second = np.argmax(hist_noFirstPeak)

    if (index_first < index_second):
        return index_first, index_second
    else:
        return index_second, index_first

def debug():
    img_path = "./flatfieldBinary.png"
    ignoreRegion = 80  # how many pixel rows to be ignore around the highest value pixel

    img = cv2.imread(img_path)
    img_gray = img  # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = historgramOnYAxis(img_gray)
    index_first = np.argmax(hist)
    hist_noFirstPeak = hist.copy()
    hist_noFirstPeak[index_first - ignoreRegion: index_first + ignoreRegion] = 0
    index_second = np.argmax(hist_noFirstPeak)

    x = range(0, img_gray.shape[1])

    fig, ax = plt.subplots(2, sharex='col', sharey='row')

    ax[0].plot(x, hist)
    ax[0].set(xlabel="rows", ylabel="mean pixel value of the row", title="Histogram")

    ax[1].plot(x, hist_noFirstPeak)
    ax[1].set(xlabel="rows", ylabel="mean pixel value of the row",
              title="Histogram without" + str(ignoreRegion) + "highest peak")
    plt.show()

# debug()
