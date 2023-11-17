import torch
import cv2
import numpy as np
from torchvision import transforms
from nets.model import unt_rdefnet34, unt_sdefnet18, unt_srdefnet18, unt_rdefnet18
from utils.palette import palette


# -----------------------------------
# device
# -----------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


weights = ''
model = unt_srdefnet18(weights)
model = model.to(device)
model.eval()

img = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/val2017/000000000139.jpg'
img = cv2.imread(img)

img[:, :, ::1] = img[:, :, ::-1]

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t = torch.unsqueeze(img_t, dim=0)
img_t.requires_grad = True
img_t = img_t.to(device)

output = model(img_t)

img[:, :, ::1] = img[:, :, ::-1]
pred = palette(output)
pred = cv2.addWeighted(img, 0.5, pred, 0.3, 50)
cv2.imshow('pred', pred)
cv2.waitKey(0)
