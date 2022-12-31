import numpy as np
from utils import chi2dist
from scipy.spatial.distance import cdist

def get_image_distance(hist1, histSet, method):
    # HIST1: 1 x K (K = # of clusters)
    # HISTSET: M x K (M = number of images, each image has 1 x K)
    # RETURN: 1 x M
    M, K = histSet.shape
    # -----fill in your implementation here --------
    if method == 'euclidean':
        dist = cdist(hist1, histSet, metric = 'euclidean')
    else:
        dist = np.apply_along_axis(lambda r : chi2dist(hist1, r.reshape(K, 1).T), 1, histSet)
        # print('chis distances:', np.min(dist), dist.shape)
    # ----------------------------------------------

    return dist
