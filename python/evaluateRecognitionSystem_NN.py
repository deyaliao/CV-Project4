import numpy as np
import pickle
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words
from getImageDistance import get_image_distance
import cv2 as cv
from createFilterBank import create_filterbank
import skimage.color



# -----fill in your implementation here --------
trainTest = pickle.load(open('../data/traintest.pkl', 'rb'))
visionRandom = pickle.load(open('../data/visionRandom.pkl', 'rb'))
visionHarris = pickle.load(open('../data/visionHarris.pkl', 'rb'))


dictionaryR, dictionaryH = visionRandom['dictionary'], visionHarris['dictionary']
filterBank = visionRandom['filterBank']
trainFeaturesR, trainFeaturesH = visionRandom['trainFeatures'], visionHarris['trainFeatures']

trainLabels = visionRandom['trainLabels']
testLabels = trainTest['test_labels']

imageNames = trainTest['test_imagenames']
K, n3 = dictionaryR.shape
total = len(imageNames)
print('dimension check:', K, trainFeaturesR.shape, trainFeaturesH.shape)
metricREuc, metricRChi, metricHEuc, metricHChi = 0, 0, 0, 0
confusionREuc, confusionRChi, ConfusionHEuc, ConfusionHChi = np.zeros((8,8)), np.zeros((8,8)), np.zeros((8,8)), np.zeros((8,8))
# Classify each test_imagenames, how... 

print(testLabels)
# print(create_filterbank().shape)
for i, image in enumerate(imageNames):
    print(i, 'image processed')
    image = cv.imread('../data/' + image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    wordMapR = get_visual_words(image, dictionaryR, filterBank)
    wordMapH = get_visual_words(image, dictionaryH, filterBank)

    cv.imwrite('testingMap.jpg', 255*skimage.color.label2rgb(wordMapR))
    cv.imwrite('testingMap.jpg', 255*skimage.color.label2rgb(wordMapH))

    histR = get_image_features(wordMapR, K)
    histH = get_image_features(wordMapH, K)

    distREuc = np.argmin(get_image_distance(histR, trainFeaturesR, "euclidean"))
    distRChi = np.argmin(get_image_distance(histR, trainFeaturesR, "chi2"))

    distHEuc = np.argmin(get_image_distance(histH, trainFeaturesH, "euclidean"))
    distHChi = np.argmin(get_image_distance(histH, trainFeaturesH, "chi2"))

    # distHCHI: u have distances against T images
    REucLabel = int(trainLabels[distREuc] - 1)
    RChiLabel = int(trainLabels[distRChi] - 1)
    HEucLabel = int(trainLabels[distHEuc] - 1)
    HChiLabel = int(trainLabels[distHChi] - 1)

    trueLabel = int(testLabels[i] - 1)
    if trueLabel == REucLabel:
        metricREuc += 1
    confusionREuc[trueLabel][REucLabel] += 1
    
    if trueLabel == RChiLabel:
        metricRChi += 1
    confusionRChi[trueLabel][RChiLabel] += 1

    if trueLabel == HEucLabel:
        metricHEuc += 1
    ConfusionHEuc[trueLabel][HEucLabel] += 1

    if trueLabel == HChiLabel:
        metricHChi += 1
    ConfusionHChi[trueLabel][HChiLabel] += 1
        
metricREuc, metricRChi, metricHEuc, metricHChi = metricREuc/total, metricRChi/total, metricHEuc/total, metricHChi/total

print('Random, Euclidean', metricREuc, confusionREuc)
print('Random, Chi', metricRChi, confusionRChi)
print('Harris, Euclidean', metricHEuc, ConfusionHEuc)
print('Harris, Chi', metricHChi, ConfusionHChi)

# ----------------------------------------------
