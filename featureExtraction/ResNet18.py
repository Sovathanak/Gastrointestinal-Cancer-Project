import torch
import torchvision.models as models

# resnet18_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# resnet18_model.eval()

# For further steps: https://pytorch.org/hub/pytorch_vision_resnet/

# The method above does not work properly for me, I researched further and found this instead, it works for me but just make sure
# that you install the full pytorch package
# use this: pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

"""Model creation"""
resnet18 = models.resnet18(pretrained=True)
resnet18 = resnet18.cpu()
# print(resnet18)

# Below feature extractor is taken from https://stackoverflow.com/questions/55083642/extract-features-from-last-hidden-layer-pytorch-resnet18
# See the answer by Manoj Mohan (bottommost post)

"""Strip last layer of NN (which hold features)"""
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])
# To use this, just call: output = feature_extractor(input), and var output will contain the features

"""Test"""
x = torch.randn([1, 3, 224, 224])  # Random input
output = feature_extractor(x)  # This holds the features corresponding to input x
# print(output.shape)