import numpy as np
import pickle
from createFilterBank import create_filterbank
from getImageFeatures import get_image_features


# -----fill in your implementation here --------
# Each pickle: store a dictionary with multiple things, like a data structure dictionary? 
# Goal1: save visionRandom.pkl + visionHarris.pkl 
dictionaryH = pickle.load(open('../data/dictionaryHarris.pkl', 'rb'))
dictionaryR = pickle.load(open('../data/dictionaryRandom.pkl', 'rb'))
K, n3 = dictionaryH.shape
# Call filterBank again, save it 
filterBank = create_filterbank()
# trainFeatures (get pickle from the previous function (svae it first)) – one for harris, one for random! 
training = pickle.load(open('../data/traintest.pkl', 'rb'))
imageNames = training['train_imagenames']
T = len(imageNames)

# HARRIS: these wordmaps are definitely generated..
wordMapsHarris = ['../data/' + s.replace('.jpg', '_Harris.pkl') for s in imageNames]

# RANDOM
wordMapsRandom = ['../data/' + s.replace('.jpg', '_Random.pkl') for s in imageNames]

trainFeaturesHarris, trainFeaturesRandom = np.empty((T, K)), np.empty((T, K))
for i in range(T):
    print('image ', i, ' processed')
    wmH = pickle.load(open(wordMapsHarris[i], 'rb'))
    wmR = pickle.load(open(wordMapsRandom[i], 'rb'))
    histH = get_image_features(wmH, K) #1 x K dimension, these are hist sets
    histR = get_image_features(wmR, K)
    trainFeaturesHarris[i] = histH
    trainFeaturesRandom[i] = histR

trainLabels = training['train_labels']
print(trainLabels.shape)

# ----------------------------------------------

visionRandom = {'dictionary': dictionaryR, 'filterBank': filterBank, 'trainFeatures': trainFeaturesRandom, 'trainLabels': trainLabels}
visionHarris = {'dictionary': dictionaryH, 'filterBank': filterBank, 'trainFeatures': trainFeaturesHarris, 'trainLabels': trainLabels}

pickle.dump(visionRandom, open('../data/visionRandom.pkl', 'wb'))
pickle.dump(visionHarris, open('../data/visionHarris.pkl', 'wb'))