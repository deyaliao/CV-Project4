import numpy as np
import pickle
import cv2 as cv
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
import skimage.color

def get_dist(a, b):
    return cdist(a, b, metric = 'euclidean')


def get_visual_words(I, dictionary, filterBank):
    # I : H x W x 3
    # dictionary: K x 3n
    # filterBank: n
    # RESULT WORD MAP: H x W
    # -----fill in your implementation here --------
    r, c, chan = I.shape
    n3 = 3*len(filterBank)
    filteredI = extract_filter_responses(I, filterBank).reshape((r * c, n3))
    
    wordMap = cdist(filteredI, dictionary, metric='euclidean')
    
    # Use argmin to get the smallest numbers
    minimums = np.argmin(wordMap, axis = 1)

    wordMap = minimums.reshape((r, c))

    # ----------------------------------------------

    return wordMap


# Load two dictionaries... using pickle 

# Pass in images, dictionary, random/harris --> 12 total 

# # Campus class
# im1H = pickle.load(open('../data/campus/sun_btvujmyjnlixtnfb_Harris.pkl', 'rb'))
# im1R = pickle.load(open('../data/campus/sun_btvujmyjnlixtnfb_Random.pkl', 'rb'))
# im2H = pickle.load(open('../data/campus/sun_dmdmbcmegrqlcguu_Harris.pkl', 'rb'))
# im2R = pickle.load(open('../data/campus/sun_dmdmbcmegrqlcguu_Random.pkl', 'rb'))
# im3H = pickle.load(open('../data/campus/sun_dzoonyvurvuvxkul_Harris.pkl', 'rb'))
# im3R = pickle.load(open('../data/campus/sun_dzoonyvurvuvxkul_Random.pkl', 'rb'))

# # AIRPORT CLASS
# im4H = pickle.load(open('../data/airport/sun_aiwqtdtnwqluftlt_Harris.pkl', 'rb'))
# im4R = pickle.load(open('../data/airport/sun_aiwqtdtnwqluftlt_Random.pkl', 'rb'))
# im5H = pickle.load(open('../data/airport/sun_afijvcdfbowiakrd_Harris.pkl', 'rb'))
# im5R = pickle.load(open('../data/airport/sun_afijvcdfbowiakrd_Random.pkl', 'rb'))
# im6H = pickle.load(open('../data/airport/sun_aihcswareotembdr_Harris.pkl', 'rb'))
# im6R = pickle.load(open('../data/airport/sun_aihcswareotembdr_Random.pkl', 'rb'))


# cv.imwrite('3.1im4H.jpg', 255*skimage.color.label2rgb(im4H))
# cv.imwrite('3.1im4R.jpg', 255*skimage.color.label2rgb(im4R))
# cv.imwrite('3.1im5H.jpg', 255*skimage.color.label2rgb(im5H))
# cv.imwrite('3.1im5R.jpg', 255*skimage.color.label2rgb(im5R))
# cv.imwrite('3.1im6H.jpg', 255*skimage.color.label2rgb(im6H))
# cv.imwrite('3.1im6R.jpg', 255*skimage.color.label2rgb(im6R))
