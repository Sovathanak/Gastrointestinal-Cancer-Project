"""This is Alex's implementation of PCA. Some code adapted from FIT3181 Tutorial 2"""
# Load required packages
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import os
from PIL import Image as PImage

# Import images
path = "C:\\Users\\Alex Waddington\\OneDrive\\Documents\\GitHub\\Gastrointestinal-Cancer-Project\\images"
for dirname, _, filename in os.walk("/Gastrointestinal-Cancer-Project/images"):
    print(dirname, _, filename)

"""def load_images(path):
    file_list = os.listdir(path)
    file_path_MSIMUT = path + "\\" + file_list[0]
    file_path_MSS = path + "\\" + file_list[1]
    image_list_MSIMUT = os.listdir(file_path_MSIMUT)
    image_list_MSS = os.listdir(file_path_MSS)
    x_MSS = []
    x_MSIMUT = []
    for image in image_list_MSIMUT[0:20]:
        img = PImage.open(file_path_MSIMUT + "\\" + image)
        x_MSIMUT.append(img)
        img.close()
    for image in image_list_MSS[0:20]:
        img = PImage.open(file_path_MSS + "\\" + image)
        x_MSS.append(img)
        img.close()
    return x_MSIMUT, x_MSS"""


"""images = load_images(path)
x_MSIMUT = images[0]
x_MSS = images[1]
"""
