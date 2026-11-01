import timm
import torch
import torchvision.transforms as T

from PIL import Image
from urllib.request import urlopen

model = timm.create_model("hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True)
model = model.eval()

train_transforms = T.Compose([T.Resize(224), 
                              T.ToTensor(), 
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

output = model(train_transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor
# output is a (1, num_features) shaped tensor
