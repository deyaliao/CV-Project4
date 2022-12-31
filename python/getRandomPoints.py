import numpy as np
import cv2 as cv
import random
from matplotlib import image
import matplotlib.pyplot as plt

def get_random_points(I, alpha):
    # Matrix of alpha x 2 of random pixel locations (x,y)
    r, c, colors = I.shape
    xs, ys = np.reshape(np.random.choice(r, alpha), (alpha, 1)), np.reshape(np.random.choice(c, alpha), (alpha, 1))
    points = np.concatenate((xs, ys), axis = 1)
    return points


# im1 = image.imread('../data/airport/sun_aerinlrdodkqnypz.jpg')
# im2 = image.imread('../data/campus/sun_abslhphpiejdjmpz.jpg')
# im3 = image.imread('../data/landscape/sun_abvvlqznpdszhjnh.jpg')

# im1RP = get_random_points(im1, 500)
# im2RP = get_random_points(im2, 500)
# im3RP = get_random_points(im3, 500)

# plt.scatter(im3RP[:, 0], im3RP[:, 1], s = 5)
# plt.imshow(im3)
# plt.show()
