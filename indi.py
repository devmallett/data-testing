import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# python indi.py

DATADIR = "C:/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        # img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)
        img_array = cv2.imread(os.path.join(path, img))
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break
print(img_array.shape)