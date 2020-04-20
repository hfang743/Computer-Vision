import cv2
import numpy as np
import glob
import os


def grayscale(img):
    # plt.imshow(gray, cmap='gray')
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):   #Applies the Canny transform
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):  #Applies a Gaussian Noise kernel
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices): #Applies an image mask.
    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
#     """
#     NOTE: this is the function you might want to use as a starting point once you want to
#     average/extrapolate the line segments you detect to map out the full
#     extent of the lane.
# ​
#     Think about things like separating line segments by their
#     slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
#     line vs. the right line.  Then, you can average the position of each of
#     the lines and extrapolate to the top and bottom of the lane.
# ​
#     This function draws `lines` with `color` and `thickness`.
#     Lines are drawn on the image inplace (mutates the image).
#     If you want to make the lines semi-transparent, think about combining
#     this function with the weighted_img() function below
#     """
#     #     for line in lines:
#     #         for x1,y1,x2,y2 in line:
#     #             cv2.line(img, (x1, y1), (x2, y2), color, thickness)
#     right_x = []
#     right_y = []
#     left_x = []
#     left_y = []
#     left_slope = []
#     right_slope = []
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             slope = ((y2 - y1) / (x2 - x1))
#             if slope >= 0.2:
#                 # right_slope.extend(int(slope))
#                 right_x.extend((x1, x2))
#                 right_y.extend((y1, y2))
#
#             elif slope <= -0.2:
#                 # left_slope.extend(int(slope))
#                 left_x.extend((x1, x2))
#                 left_y.extend((y1, y2))
#     right_fit = np.polyfit(right_x, right_y, 1)
#     right_line = np.poly1d(right_fit)
#     x1R = 550
#     y1R = int(right_line(x1R))
#     x2R = 850
#     y2R = int(right_line(x2R))
#     cv2.line(img, (x1R, y1R), (x2R, y2R), color, thickness)
#     left_fit = np.polyfit(left_x, left_y, 1)
#     left_line = np.poly1d(left_fit)
#     x1L = 120
#     y1L = int(left_line(x1L))
#     x2L = 425
#     y2L = int(left_line(x2L))
#     cv2.line(img, (x1L, y1L), (x2L, y2L), color, thickness)
#
#
# def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
#     """
#     `img` should be the output of a Canny transform.
# ​
#     Returns an image with hough lines drawn.
#     """
#     lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
#                             maxLineGap=max_line_gap)
#     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     draw_lines(line_img, lines)
#     return line_img
#
#
# # Python 3 has support for cool math symbols.
#
# def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
#     """
#     `img` is the output of the hough_lines(), An image with lines drawn on it.
#     Should be a blank image (all black) with lines drawn on it.
# ​
#     `initial_img` should be the image before any processing.
# ​
#     The result image is computed as follows:
# ​
#     initial_img * α + img * β + λ
#     NOTE: initial_img and img must be the same shape!
#     """
#     return cv2.addWeighted(initial_img, α, img, β, λ)


def pipeline(input_image):
    image = input_image


    # os.listdir("./data/")
    # gray = grayscale(image)
    # Gaussian smoothing
    kernel_size = 5
    gau = gaussian_blur(input_image, kernel_size)

    # Canny
    low_threshold = 100
    high_threshold = 255
    edges = canny(gau, low_threshold, high_threshold)

    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (150, 330), (800, 250), (imshape[1], imshape[0])]], dtype=np.int32)
    region = region_of_interest(edges, vertices)

    # rho = 1  # distance resolution in pixels of the Hough grid
    # theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    # min_line_len = 20  # minimum number of pixels making up a line
    # max_line_gap = 20
    #
    #
    #
    # line_img = hough_lines(region, rho, theta, threshold, min_line_len, max_line_gap)
    # line_last = weighted_img(line_img, image, α=0.8, β=1., λ=0.)
    return region


# from moviepy.editor import VideoFileClip
# from IPython.display import HTML


def process_image(image):
    result = pipeline(image)
    return result




#
#
# convert_image_path = './data/'
# fps = 24
# size = (1392,512)
# videoWriter = cv2.VideoWriter('./Video1_test.mp4',cv2.VideoWriter_fourcc('I','4','2','0'),
#                               fps,size)
# for img in glob.glob(convert_image_path + "/*.png") :
#     read_img = cv2.imread(img)
#     videoWriter.write(read_img)
# videoWriter.release()
#
#
# white_output = 'Video1_test.mp4'
# clip1 = VideoFileClip("Video1_test.mp4")
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(white_output, audio=False)
# clip1.preview()
