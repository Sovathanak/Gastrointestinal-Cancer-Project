import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn import decomposition
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# The tensorflow/keras version is not working, so it has been removed

# This function needs to have scipy installed (pip3 install scipy)

BATCH_SIZE = 10

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
inception = models.inception_v3(pretrained=True)
inception = inception.to(device)


# print(inception)

# Implementation below is adapted from:
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119
# See post #49 by fmassa

class MyInceptionFeatureExtractor(nn.Module):
    def __init__(self, inception, transform_input=False):
        super(MyInceptionFeatureExtractor, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = inception.avgpool
        self.dropout = inception.dropout

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = self.maxpool1(x)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = self.maxpool2(x)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        return x


feature_extractor = MyInceptionFeatureExtractor(inception)

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

    # Decomposition of nx*ny features and choosing only the 10 most valuable features to train on
    pca = decomposition.PCA(n_components=10)
    pcaFeature = pca.fit_transform(features)
    
    # if you need to see the format/structure after using the pca, uncomment the 2 lines below
    # print(pcaX)
    # break

    labels = labels.detach().numpy()
    label_temp = []
    for x in labels:
        if x == 0:
            label_temp.append("MSIMUT")
        elif x == 1:
            label_temp.append("MSS")
    
    df = pd.DataFrame(pcaFeature, columns=['Component1', 'Component2', 'Component3', 'Component4', 'Component5', 'Component6', 'Component7', 'Component8', 'Component9', 'Component10'])
    df['Cancer'] = label_temp
    df.to_csv("extractedFeatures/InceptionV3features.csv", mode = 'a', header=False, index=False)
