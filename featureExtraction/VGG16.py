import torch
from torch.utils import data
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

# The tensorflow/keras version is not working, so it has been removed

BATCH_SIZE = 5

"""Stores images in batches"""
# ImageFolder automatically labels images and transforms images to tensors
# DataLoader stores the images in batches
image_path = ""
image = ImageFolder(root=image_path, transform=transforms.ToTensor()) 
dataset = DataLoader(dataset=image, batch_size=BATCH_SIZE, shuffle=False)

# View first 5 images stored in dataset, taken from: https://towardsdatascience.com/how-to-apply-a-cnn-from-pytorch-to-your-images-18515416bba1
def train_imshow():
    classes = ('MSIMUT', 'MSS') # Defining the classes we have
    dataiter = iter(dataset)
    images, labels = dataiter.next()
    fig, axes = plt.subplots(figsize=(11, 4), ncols=5)
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0)) 
        ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
    plt.show()
# train_imshow()

vgg16 = models.vgg16(pretrained=True)
vgg16 = vgg16.cpu()
# print(vgg16)

# Below feature extractor is taken from https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
# See the answer by Manoj Mohan (bottommost post)

"""Strip last layer of NN (which hold features)"""
feature_extractor = torch.nn.Sequential(*list(vgg16.children())[:-1])
# To use this, just call: output = feature_extractor(input), and var output will contain the features

"""Test"""
x = torch.randn([1, 3, 224, 224])  # Random input
output = feature_extractor(x)  # This holds the features corresponding to input x
# print(output.shape)

"""Extract features from batches"""
# feature extractor only works with 4-dimensional inputs (single images are 3-dimensional inputs)
extracted_features = []
dataiter = iter(dataset)
for i in range(len(dataset)):
    images, labels = dataiter.next()
    features = feature_extractor(images)
    x = features, labels
    extracted_features.append(x)

print(extracted_features)





