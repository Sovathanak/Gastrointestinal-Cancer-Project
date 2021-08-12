import torch
from torchvision import *
from torchvision.models import resnet

resnet18_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
resnet18_model.eval()

# For further steps: https://pytorch.org/hub/pytorch_vision_resnet/

# The method above does not work properly for me, I researched further and found this instead, it works for me but just make sure
# that you install the full pytorch package
# use this: pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

resnet18 = models.resnet18(pretrained=True)
resnet18 = resnet18.cuda()
print(resnet18)