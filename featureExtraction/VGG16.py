import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from sklearn import decomposition
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# The tensorflow/keras version is not working, so it has been removed

BATCH_SIZE = 12

"""Stores images in batches"""
# ImageFolder automatically labels images and transforms images to tensors
# DataLoader stores the images in batches

# Make sure to put in the directory of your "images" folder and not the specific folders.
# put in like so: ./images/
image_path = "./images/"
image = ImageFolder(root=image_path, transform=transforms.ToTensor())
dataset = DataLoader(dataset=image, batch_size=BATCH_SIZE, shuffle=False)

# Checks if you have cuda. If yes, use cuda, else use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

"""Model creation"""
vgg16 = models.vgg16(pretrained=True)
vgg16 = vgg16.to(device)
# print(vgg16)

# Below feature extractor is taken from:
# https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
# See the answer by Manoj Mohan (bottommost post)

"""Strip last layer of NN (which hold features)"""
torch.cuda.empty_cache()
feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-1])
# To use this, just call: output = feature_extractor(input), and var output will contain the features

"""Extract features from batches and apply PCA"""
# feature extractor only works with 4-dimensional inputs (single images are 3-dimensional inputs)
dataiter = iter(dataset)

# code block which extracts the features from the images and send the output (array) to csv file
for i in range(len(dataset)):
    images, labels = dataiter.next()
    images = images.to(device)
    features = feature_extractor(images)

    # PCA process
    # reduce tensor dimensions from 4D to 1D
    features = torch.flatten(features, 1)

    # convert the torch tensor to a numpy tensor for pca, there will be an error if this line is removed
    features = features.cpu().detach().numpy()

    # Decomposition of features and choosing only the 10 most valuable features to train on
    pca = decomposition.PCA(n_components=10)
    pcaFeature = pca.fit_transform(features)

    # convert the labels into human-readable formats (0 to MSIMUT & 1 to MSS)
    labels = labels.detach().numpy()
    label_temp = []
    for x in labels:
        if x == 0:
            label_temp.append("MSIMUT")
        elif x == 1:
            label_temp.append("MSS")
    
    # create dataframe to store and format the data
    df = pd.DataFrame(pcaFeature, columns=['Component1', 'Component2', 'Component3', 'Component4', 'Component5', 'Component6', 'Component7', 'Component8', 'Component9', 'Component10'])
    df['Cancer'] = label_temp
    
    # append the formatted data stored in the dataframe to the respective csv file 
    df.to_csv("extractedFeatures/VGG16features.csv", mode = 'a', header=False, index=False)
