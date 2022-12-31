import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *
from createFilterBank import create_filterbank

# Make sure that
def scale(I):
    minVal = np.min(I)
    if minVal < 0:
        I = I + abs(minVal)
    return I * (255.0 / I.max())


# Given a filter bank and image, what are we supposed to do?
def extract_filter_responses(I, filterBank):
    n = len(filterBank)
    r, c, k = I.shape
    filterResponses = np.empty((r,c,n*3))
    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------
    # FilterBank is an array of filters:
    iLab = rgb2lab(I)
    c1, c2, c3 = iLab[:,:,0], iLab[:,:,1], iLab[:,:,2]
    for i in range(n):
        # Filterimage with this individual color channel: (rxcx1)
        filter = filterBank[i]
        i1 = imfilter(c1, filter)
        i2 = imfilter(c2, filter)
        i3 = imfilter(c3, filter)

        # print(i1, i2, i3)
        filterResponses[:, :, 3*i], filterResponses[:, :, 3*i+1], filterResponses[:, :, 3*i+2] = i1, i2, i3

    # Scale filter Responses
    # ----------------------------------------------
    
    return filterResponses


airport = cv.imread('../data/airport/sun_aerinlrdodkqnypz.jpg')
image = extract_filter_responses(airport, create_filterbank())
im1, im2, im3 = scale(image[:,:,6]), scale(image[:,:,28]), scale(image[:,:,59])

# Convert color to RGB 



cv.imwrite('im1.jpg', im1)
cv.imwrite('im2.jpg', im2)
cv.imwrite('im3.jpg', im3)

