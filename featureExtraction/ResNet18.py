import torch

resnet18_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
resnet18_model.eval()

# For further steps: https://pytorch.org/hub/pytorch_vision_resnet/
