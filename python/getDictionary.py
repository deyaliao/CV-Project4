import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv.imread('../data/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default: image path is already loaded here, now u just have to manipulate 
        
        # -----fill in your implementation here --------
        # Apply filterbank to image
        images = extract_filter_responses(image, filterBank)

        # Get alpha points for each image:
        if method == "Random":
            points = get_random_points(image, alpha) #alpha x 2 matrix
        else:
            # aren't there 60 new images essentially? wdym pass in the image
            points = get_harris_points(image, alpha, 0.05)

        for j, point in enumerate(points):
            x,y = int(point[0]), int(point[1])
            index = i*alpha + j
            pixelResponses[index] = images[x][y]


        # ----------------------------------------------

    dictionary = KMeans(n_clusters=K, random_state=0, algorithm='elkan').fit(pixelResponses).cluster_centers_
    return dictionary

