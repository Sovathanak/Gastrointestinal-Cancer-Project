import torch
import torchvision.models as models

# The tensorflow/keras version is not working, so it has been removed

vgg16 = models.vgg16(pretrained=True)
vgg16 = vgg16.cuda() if torch.cuda.is_available() else vgg16.cpu()
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