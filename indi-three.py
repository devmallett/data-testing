import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2


DATADIR = "C:/Datasets/PetImages"
IMAGES = ["Cat", "Dog"]


# ONce loaded to our program, we have to iterate through 
# the images and assign digits to the numbers 
#

# path = os.path.join(DATADIR)

# print(path)


 for some_iamge in IMAGES:
    # locating the path of the image
    path = os.path.join(DATADIR, some_iamge)
    for images in os.listdir(path):
        set_array = cv2.imread(os.path.join(path, images), cv2.IMREAD_GRAYSCALE)
        

        

