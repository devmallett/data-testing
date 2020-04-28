import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
# python indi.py

# DATADIR = "C:/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        # img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(img_array, cmap="gray")
        # plt.show()
        break
    break
# print(img_array.shape)

IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE,1 ))
plt.imshow(new_array, cmap = 'gray')
# plt.show()


# Crating training data set 

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        # img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)
        try:
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass
create_training_data()


# print(len(training_data))

# balance your  data set  

# shuffle data 
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# always want to have this data
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
