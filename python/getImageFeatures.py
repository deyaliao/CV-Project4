import numpy as np

def get_image_features(wordMap, dictionarySize):
    # wordMap: H x W  
    # dictionarySize: K
    # return: 1 x K

    # -----fill in your implementation here --------
    h = np.empty((1, dictionarySize))
    for i in range(dictionarySize):
        h[0][i] = np.count_nonzero(wordMap == i)

    # Normalize histogram
    h = h / np.sum(np.abs(h))
    # ----------------------------------------------
    
    return h
