import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter
from matplotlib import image
import matplotlib.pyplot as plt


def get_harris_points(I, alpha, k):
    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------

    # Compute X and Y gradients for image (xx, xy) should be computed, use Sobel
    # imgx: convolve smoothed image with x-oriented sobel filter
    x_sobel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    gradX = imfilter(I, x_sobel) #not sure if I'm convolving this correctly

    # imgy: convolve smoothed image with the y-oriented sobel filter 
    y_sobel = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
    gradY = imfilter(I, y_sobel)

    gradient_array = np.degrees(np.arctan2(gradX, gradY)) % 180
    amplitude = np.sqrt(np.square(gradX) + np.square(gradY))

    # Get four image gradient matrices:
    xx = np.multiply(gradX, gradX)
    xy = np.multiply(gradX, gradY)
    yx = np.multiply(gradY, gradX)
    yy = np.multiply(gradY, gradY)

    # Compute M for every single point 
    r, c = I.shape
    topResponses, points = np.zeros((alpha)), np.zeros((alpha, 2))
    xxP = np.pad(xx, ((2,2), (2,2)), "edge") #padded image
    xyP = np.pad(xy, ((2,2), (2,2)), "edge")
    yxP = np.pad(yx, ((2,2), (2,2)), "edge")
    yyP = np.pad(yy, ((2,2), (2,2)), "edge")
    for i in range(r):
        for j in range(c):
            sumXX = np.sum(xxP[i:i+5, j:j+5])
            sumXY = np.sum(xyP[i:i+5, j:j+5])
            sumYX = np.sum(yxP[i:i+5, j:j+5])
            sumYY = np.sum(yyP[i:i+5, j:j+5])
            m = np.array([[sumXX, sumXY], [sumYX, sumYY]])
            res = np.linalg.det(m) - k*np.trace(m)

            # Update r values
            if res > np.amin(topResponses):
                index = np.argmin(topResponses)
                topResponses[index] = res
                points[index] = [i,j]
                # print('after putting', res, points[index], i, j)

    # ----------------------------------------------
    return points


# im1 = image.imread('../data/airport/sun_aerinlrdodkqnypz.jpg')
# im2 = image.imread('../data/campus/sun_abslhphpiejdjmpz.jpg')
# im3 = image.imread('../data/landscape/sun_abvvlqznpdszhjnh.jpg')

# im1RP = get_harris_points(im1, 500, 0.05)
# im2RP = get_harris_points(im2, 500, 0.05)
# im3RP = get_harris_points(im3, 500, 0.05)
# plt.scatter(im2RP[:, 1], im2RP[:, 0], s = 3)
# plt.imshow(im2)
# plt.show()
