import torch
import cv2
import numpy as np
from torchvision import transforms
from nets.model import unt_rdefnet34, unt_sdefnet18, unt_srdefnet18, unt_rdefnet18
from utils.palette import palette
from utils.score import loss_bce
from data.dataset import dataloader
from torchvision.transforms import v2

# -----------------------------------
# device
# -----------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    weights = 'checkpoints/ep2-loss0.0485747023217943.pth'
    model = unt_rdefnet34(num_classes=91, pretrained_own=weights)
    model = model.to(device)
    model.eval()
    hw = (400, 400)
    val_img_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/val2017'
    val_label_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/label_val/id_labels'
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        v2.Resize(hw, antialias=True)
    ])
    target_transform = v2.Compose([
        v2.Resize(hw, antialias=True)
    ])
    target_info = (90+1, hw[0], hw[1])
    val_data_loader     = dataloader(val_img_path, val_label_path, target_info, 1, 'train', transform, target_transform)
    _, target = next(iter(val_data_loader))
    print(target.shape)
    target = target.to(device)


    img = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/val2017/000000001296.jpg'
    img = cv2.imread(img)
    hw = img.shape[:2]
    img = cv2.resize(img, (400, 400))

    img[:, :, ::1] = img[:, :, ::-1]

    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img)
    img_t = torch.unsqueeze(img_t, dim=0)
    img_t.requires_grad = True
    img_t = img_t.to(device)

    output = model(img_t)
    loss = loss_bce(output=output, target=target, mode='val')
    print(loss)

    img[:, :, ::1] = img[:, :, ::-1]
    pred = palette(output)
    print(img.shape)
    print(pred.shape)
    pred = cv2.addWeighted(src1=img, alpha=0.4, src2=pred, beta=0.8, gamma=0.0, dtype=cv2.CV_8UC3)
    cv2.imshow('pred', pred)
    cv2.waitKey(0)
