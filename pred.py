import torch
import cv2
import numpy as np
import torchvision
from torchvision import transforms
from nets.model import unt_rdefnet34, unt_sdefnet18, unt_srdefnet18, unt_rdefnet18
from utils.palette import palette
from utils.score import loss_bce
from data.dataset import dataloader
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()

# -----------------------------------
# device
# -----------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    weights = 'checkpoints/ep70-val_loss0.15663308305200188-miou0.023155.pth'
    model = unt_rdefnet18(num_classes=91, pretrained_own=weights)
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
    val_data_loader     = dataloader(val_img_path, val_label_path, target_info, 1, 'val', transform, target_transform)
    for img, target in val_data_loader:
        target = target.to(device)

        img = img.to(device)

        output = model(img)
        loss = loss_bce(output=output, target=target, mode='val')

        pred, indxs = palette(output)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = img * 255
        img[:, :, ::1] = img[:, :, ::-1]
        pred = cv2.addWeighted(src1=img, alpha=0.5, src2=pred, beta=10, gamma=0.0, dtype=cv2.CV_8UC3)
        for i in range(0, pred.shape[0], 100):
            for j in range(0, pred.shape[1], 100):
                if int(indxs[i, j]) != 0:
                    pred = cv2.rectangle(pred, (i, j-30), (i+50, j), (0, 0, 0), -1)
                    pred = cv2.putText(pred, str(int(indxs[i, j])), (i, j), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('pred', pred)
        cv2.waitKey(0)
