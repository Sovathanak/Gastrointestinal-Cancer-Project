import torchvision.models as models

# The tensorflow/keras version is not working, so it has been removed

# This function needs to have scipy installed (pip3 install scipy)

inception = models.inception_v3(pretrained=True)
inception = inception.cpu()
# print(inception)