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

BATCH_SIZE = 12

'''Stores images in batches'''
# ImageFolder automatically labels images and transforms images to tensors
# DataLoader stores the images in batches

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Make sure to put in the directory of your 'images' folder and not the specific folders.
# put in like so: ./images/
image_path = './images/'
image = ImageFolder(root=image_path, transform = preprocess)
dataset = DataLoader(dataset=image, batch_size=BATCH_SIZE, shuffle=False)


# # Checks if you have cuda. If yes, use cuda, else use cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# '''Model creation'''
inceptionV3 = models.inception_v3(pretrained=True)
inceptionV3 = inceptionV3.to(device)

dataiter = iter(dataset)
images, labels = dataiter.next()
images = images.to(device)
with torch.no_grad():
    output = inceptionV3(images)
# # print(output[0])
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)


# # From Pytorch docs:
# # # sample execution (requires torchvision)
# from PIL import Image
# # from torchvision import transforms
# input_image = Image.open(image_path)
# preprocess = transforms.Compose([
#     transforms.Resize(299),
#     transforms.CenterCrop(299),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# # move the input and model to GPU for speed if available
# # if torch.cuda.is_available():
# #     input_batch = input_batch.to('cuda')
# #     inceptionV3.to('cuda')

# with torch.no_grad():
#   output = inceptionV3(input_batch)
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)
