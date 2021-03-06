import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow import keras
from  keras.models import Sequential
from  tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# python indi-two.py
# Users/Owner/Documents/data-testing/data-testing
# virtual environment: data-testing-JaEhqsaC

# Loading data from C drive
DATADIR = "C:/Datasets/PetImages"
CHUMPSKY = [ "Dog", "Cat" ]
# set

# Iterate through all examples of Cat and Dog Images 
for some_image in CHUMPSKY:
    path = os.path.join(DATADIR, some_image) 
    for imgages in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, imgages), cv2.IMREAD_GRAYSCALE)
        # img_array = cv2.imread(os.path.join(path, imgages))
        plt.imshow(img_array, cmap="gray")
        # plt.show()
        break
    break #breaks allow you to just run the function once to see what is going on 

# print(img_array.shape)

IMG_SIZE = 50
new_devin = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_devin, cmap="gray")
plt.show()

# print(new_devin)

training_data = []

# Setting Classification as Numbers 
def creating_training_data():
    for some_image in CHUMPSKY:
        path = os.path.join(DATADIR, some_image) 
        class_num = CHUMPSKY.index(some_image)
        for imgages in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, imgages), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num]) 
            except Exception as e:
                pass 
creating_training_data()

# print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1]) #actual image array
    
# Feeding into neural network
X = [] #images
y = [] #labels


for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #where '1' is the color
y = np.array(y)

print(len(X))
print(len(y))

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# pickle_in = 


# X = X/255.0

# model = keras.Sequential([
#     keras.layers.Conv2D((24946), (3,3), input_shape=X.shape[1:]),
#     keras.layers.Dense(24946, activation="relu"),
#     keras.layers.MaxPooling2D(pool_size=(2,2)),
#     keras.layers.Dense(1, activation="sigmoid")
# ])

# model.add(Conv2D(64), (3,3), input_shape = X.shape[1:])
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(64), (3,3))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Flatten())
# model.add(Dense(64))

# model.add(Dense(1))
# model.add(Activation("sigmoid"))

# model.compile(loss="mean_squared_error", optimizer="sgd")

# model.fit(X, y, batch_size=32, epochs=40, validation_split=0.1)







# Pick up at 6:44
# https://www.youtube.com/watch?v=j-3vuBynnOE

