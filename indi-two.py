import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2

# python indi-two.py
# Users/Owner/Documents/data-testing/data-testing

DATADIR = "C:/Datasets/PetImages"
CHUMPSKY = [ "Dog", "Cat" ]


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

print(new_devin)

# Pick up at 6:44
# https://www.youtube.com/watch?v=j-3vuBynnOE
