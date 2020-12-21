# USAGE
# python train.py --model mixed_classifier.h5

# import the necessary packages
from opencv_sudoku_solver.pyimagesearch.models import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model after training")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = 1e-3
EPOCHS = 40
BS = 128

# grab the MNIST dataset
print("[INFO] accessing MNIST...")
trainData = np.load("trainData.npy")
testData = np.load("testData.npy")
trainLabels = np.load("trainLabels.npy")
testLabels = np.load("testLabels.npy")

# convert the labels from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model = SudokuNet.build(width=28, height=28, depth=1, classes=20)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 决定是否加载 model
flag = False
model_path = "output/digit_classifier.h5"
if os.path.exists(model_path) and flag:
	model = load_model(model_path)
else:
	model = SudokuNet.build(width=28, height=28, depth=1, classes=20)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])


# train the network
print("[INFO] training network...")
H = model.fit(
	trainData, trainLabels,
	validation_data=(testData, testLabels),
	batch_size=BS,
	epochs=EPOCHS,
	verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testData)
print(classification_report(
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

# serialize the model to disk
print("[INFO] serializing digit model...")
model.save(args["model"], save_format="h5")


"""
画图
"""
import matplotlib.pyplot as plt

epochs=range(len(H.history['accuracy']))
plt.figure()
plt.plot(epochs, H.history['accuracy'], 'b', label='Training acc')
plt.plot(epochs, H.history['val_accuracy'],'r',label='Validation acc')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.savefig('model_V3.1_acc.jpg')

plt.figure()
plt.plot(epochs, H.history['loss'], 'b', label='Training loss')
plt.plot(epochs, H.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.savefig('model_V3.1_loss.jpg')
