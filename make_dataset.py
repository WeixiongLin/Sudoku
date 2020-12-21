import numpy as np


en_trainData = np.load("en_trainData.npy")
en_trainLabels = np.load("en_trainLabels.npy")
en_testData = np.load("en_testData.npy")
en_testLabels = np.load("en_testLabels.npy")

ch_trainData = np.load("ch_trainData.npy")
ch_trainLabels = np.load("ch_trainLabels.npy")
ch_testData = np.load("ch_testData.npy")
ch_testLabels = np.load("ch_testLabels.npy")

trainData = np.concatenate((en_trainData, ch_trainData))
trainLabels = np.concatenate((en_trainLabels, ch_trainLabels))
testData = np.concatenate((en_testData, ch_testData))
testLabels = np.concatenate((en_testLabels, ch_testLabels))

print("trainLabels.shape", trainLabels.shape)
print("testLabels.shape", testLabels.shape)
print("trainLabels.[0]", ch_trainLabels[0])
print("testLabels.[0]", ch_testLabels[0])
print("trainLabels.[1]", ch_trainLabels[1])
print("testLabels.[1]", ch_testLabels[1])
print("trainLabels.[2]", ch_trainLabels[2])
print("testLabels.[2]", ch_testLabels[2])

np.save("trainData.npy", trainData)
np.save("trainLabels.npy", trainLabels)
np.save("testData.npy", testData)
np.save("testLabels.npy", testLabels)
