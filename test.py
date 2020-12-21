import tensorflow as tf
import random
import pathlib
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

"""
超参数
"""
# initialize the initial learning rate, number of epochs to train
# for, and batch size
INIT_LR = 1e-3
EPOCHS = 1
BS = 64
Classes = 20
model_path = "opencv_sudoku_solver/output/digit_classifier.h5"

"""
1
"""
# Train Data
train_data_path = pathlib.Path("dataset/training")
train_image_paths = list(train_data_path.glob('*/*'))  
train_image_paths = [str(path) for path in train_image_paths]  # 所有图片路径的列表
random.shuffle(train_image_paths)  # 打散

# Test Data
test_data_path = pathlib.Path("dataset/testing")
test_image_paths = list(test_data_path.glob('*/*'))  
test_image_paths = [str(path) for path in test_image_paths]  # 所有图片路径的列表
random.shuffle(test_image_paths)  # 打散


image_count = len(train_image_paths)
# print(all_image_paths)
# print("image_count", image_count)

"""
2
"""
# print("train_image_paths[:5]", train_image_paths[:5])
# print("test_image_paths[:5]", test_image_paths[:5])
label_names = sorted(item.name for item in train_data_path.glob('*/') if item.is_dir())
label_to_index = dict((str(i+9), i+9) for i in range(1, 11))
print("label_to_index", label_to_index)

train_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_image_paths]
for image, label in zip(train_image_paths[:5], train_image_labels[:5]):
    print(image, ' --->  ', label)

test_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in test_image_paths]
for image, label in zip(test_image_paths[:5], test_image_labels[:5]):
    print(image, ' --->  ', label)

ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))

"""
3
"""

trainData = []
testData = []
for path in train_image_paths:
    img = cv2.imread(path, 0)
    ret, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    img = np.resize(img, (28, 28, 1))
    trainData.append(img)

for path in test_image_paths:
    img = cv2.imread(path, 0)
    ret, img = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY_INV)
    img = np.resize(img, (28, 28, 1))
    testData.append(img)

trainData = np.array(trainData, dtype='float32')
testData = np.array(testData, dtype='float32')

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0


"""
4
"""
# for i in range(21, 30):
#     cv2.imwrite("tmp/"+str(i)+".png", testData[i])
# print(trainData.shape)
# print(testData.shape)

N = len(train_image_labels)
trainLabels = np.zeros(N, dtype='int16')
for i in range(N):
    trainLabels[i] = train_image_labels[i]
print(trainLabels.shape)


N = len(test_image_labels)
testLabels = np.zeros(N, dtype='int16')
for i in range(N):
    testLabels[i] = test_image_labels[i]
print(testLabels.shape)
print("testLabels[0]", testLabels[0])

# 保存 Data, Labels 为数组
print("trainLabels.shape", trainLabels.shape)
np.save("ch_trainData.npy", trainData)
np.save("ch_trainLabels.npy", trainLabels)
np.save("ch_testData.npy", testData)
np.save("ch_testLabels.npy", testLabels)

# convert the labels from integers to vectors
le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

"""
5
"""

from opencv_sudoku_solver.pyimagesearch.models import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os


opt = Adam(lr=INIT_LR)

# 决定是否加载 model
flag = False

if os.path.exists(model_path) and flag:
	model = load_model(model_path)
else:
	model = SudokuNet.build(width=28, height=28, depth=1, classes=20)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

# train the network
print("[INFO] training network...")

cv2.imwrite("ex.png", np.resize(trainData[1], (28, 28)))
print(trainLabels[1])

history = model.fit(
	trainData, trainLabels,
	validation_data=(testData, testLabels),
	batch_size=BS,
	epochs=EPOCHS,
	verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testData)
# print(classification_report(
# 	testLabels.argmax(axis=1),
# 	predictions.argmax(axis=1),
# 	target_names=[str(x) for x in le.classes_]))
print(classification_report(
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1)))
# serialize the model to disk
print("[INFO] serializing digit model...")
model.save(model_path, save_format="h5")


"""
6
"""
import matplotlib.pyplot as plt

epochs=range(len(history.history['accuracy']))
plt.figure()
plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
# plt.plot(epochs,history.history['val_acc'],'r',label='Validation acc')
plt.title('Traing and Validation accuracy')
plt.legend()
plt.savefig('model_V3.1_acc.jpg')

plt.figure()
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
# plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.savefig('model_V3.1_loss.jpg')
