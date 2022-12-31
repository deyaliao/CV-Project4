import numpy as np
import pickle
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words
from getImageDistance import get_image_distance
import cv2 as cv
from createFilterBank import create_filterbank
import matplotlib.pyplot as plt


# -----fill in your implementation here --------
trainTest = pickle.load(open('../data/traintest.pkl', 'rb'))
visionRandom = pickle.load(open('../data/visionRandom.pkl', 'rb'))
visionHarris = pickle.load(open('../data/visionHarris.pkl', 'rb'))
imageNames = trainTest['test_imagenames']


dictionaryR, dictionaryH = visionRandom['dictionary'], visionHarris['dictionary']
filterBank = visionRandom['filterBank']
trainFeaturesR, trainFeaturesH = visionRandom['trainFeatures'], visionHarris['trainFeatures']

trainLabels = visionRandom['trainLabels']
testLabels = trainTest['test_labels']

T = len(imageNames)
train = len(trainLabels)
K, n3 = dictionaryR.shape


# Comment out Random v Harris for as little code as possible! 
allDistEuc, allDistChi = np.empty((T, train)), np.empty((T,train))
for i, image in enumerate(imageNames):
    image = cv.imread('../data/' + image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    wordMapR = get_visual_words(image, dictionaryH, filterBank)
    histR = get_image_features(wordMapR, K)
    # allDistEuc[i] = get_image_distance(histR, trainFeaturesR, "euclidean")
    allDistChi[i] = get_image_distance(histR, trainFeaturesH, "chi2")


# number of neighbors to keep track of: take the k closest points of all the distances, the smallest k distances of distance matrix
# Keep track of accuracy: the highest classifed one that's correct. 
# Get the closest distnaces (indices), get their labels, count the argmax for the label, and classify 
def EUC(distances, k):
    smallestKEuc = np.argpartition(distances, k)[:k]
    labelsEuc = [trainLabels[i] for i in smallestKEuc]
    lEuc = np.bincount(labelsEuc).argmax() #this represents the label
    return lEuc

def CHI(distances, k):
    smallestKChi = np.argpartition(distances, k)[:k] #these are all indices! 
    labelsChi = [trainLabels[i] for i in smallestKChi]
    lChi = np.bincount(labelsChi).argmax()
    return lChi

neighbors = 0
accuracies = list(range(40))
highestAccuracy = 0
ourLabel = list(range(T))
for k in range(1, 41):
    print('k test', k)
    chiLabels = np.apply_along_axis(lambda r : CHI(r, k), 1, allDistChi) #allDistChi is T x train (160 x 1331); chi labels should be 160 x 1
    print('chi labels', chiLabels, chiLabels.shape)
    print('test labels', testLabels)

    checks = chiLabels == testLabels
    correct = np.count_nonzero(checks)
    accuracy = correct/T
    print('accuracy', accuracy)
    if accuracy > highestAccuracy:
        highestAccuracy = accuracy
        ourLabel = chiLabels
        neighbors = k
    accuracies[k-1] = accuracy

# ----------------------------------------------
# Compute confusion matrix
confusionM = np.zeros((8,8))
for k in range(T):
    i, j = int(testLabels[k]), int(ourLabel[k])
    confusionM[i-1][j-1] += 1

print('optimal k', neighbors)
print(confusionM)
# Plot Graph
plt.bar(list(range(1,41)), accuracies)
plt.show()