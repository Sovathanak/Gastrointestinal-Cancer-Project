import torchvision.models as models

# The tensorflow/keras version is not working, so it has been removed

vgg16 = models.vgg16()
vgg16 = vgg16.cpu()
print(vgg16)