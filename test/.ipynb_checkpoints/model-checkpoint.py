import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms

model = models.resnet50(weights='DEFAULT')
for name, module in model.named_modules():
    print(name, ":", module)