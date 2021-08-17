import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import csv
import numpy as np
from sklearn import decomposition

# The tensorflow/keras version is not working, so it has been removed

BATCH_SIZE = 1
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=10000)

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

# Below feature extractor is taken from https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
# See the answer by Manoj Mohan (bottommost post)

"""Strip last layer of NN (which hold features)"""
torch.cuda.empty_cache()
feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-1])
# To use this, just call: output = feature_extractor(input), and var output will contain the features

"""Extract features from batches and apply PCA"""
# feature extractor only works with 4-dimensional inputs (single images are 3-dimensional inputs)
dataiter = iter(dataset)

# code block which extracts the features from the images and send the output (array) to csv file
with open("extractedFeatures/VGG16features.csv", "w") as file:
    write = csv.writer(file)
    for i in range(len(dataset)):
        images, labels = dataiter.next()
        images = images.to(device)
        features = feature_extractor(images)
        
        # PCA process
        batch_size, nsamples, nx, ny = features.shape
        
        # reshaping the dimensions of the feature tensors to 2 dimensions instead of 4
        features = features.reshape((nsamples, nx*ny))
        # print(features.shape)  # torch.Size([1, 512, 7, 7]) == [batch_size, nsamples (number of nx*ny arrays), nx, ny]

        # convert the torch tensor to a numpy tensor for pca, there will be an error if this line is removed
        features = features.cpu().detach().numpy()
        
        # Decomposition of nx*ny features and choosing only the 10 most valuable features to train on
        pca = decomposition.PCA(n_components=10)
        pcaFeature = pca.fit_transform(features)
        # if you need to see the format/structure after using the pca, uncomment the 2 lines below
        # print(pcaX)
        # break
        write.writerow((labels, pcaFeature)) # write to the csv file 