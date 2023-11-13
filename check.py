import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import datetime
from tqdm import tqdm
from nets2.defnet import defnet18, defnet50
from torchvision import transforms
from nets2.model import unt_defnet18, unt_defnet50

model = unt_defnet18()

device = (torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu')
)
model = model.to(device=device)

# -----------------------------------
# optimizer
# -----------------------------------
lr_rate = 0.001
milestones = [20, 30, 35]
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr_rate)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)

# -----------------------------------
# data_loader
# -----------------------------------
to_tensor = transforms.ToTensor()
# input = cv2.imread('Gaming_5000x3125.jpg')
# input = cv2.resize(input, (100, 100))
input = torch.rand([100, 100])
input = to_tensor(input)
input = input.unsqueeze(0)
input = input.to(device=device)

model.eval()
output = model(input)
for i in range(len(output)):
    print(output[i].shape)