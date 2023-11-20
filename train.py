import torch
import torch.nn as nn
import torchvision
import cv2
import os
import numpy as np
import datetime
from tqdm import tqdm
from utils.fit_one_epoch import fit_one_epoch
from utils.callbacks import LossHistory
from nets.model import unt_srdefnet18, unt_rdefnet18, unt_rdefnet34, unt_sdefnet18
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

if __name__=='__main__':
    # -----------------------------------
    # model
    # -----------------------------------
    num_classes = 90
    model = unt_rdefnet18(num_classes=num_classes+1, pretrained_own=False)

    model = model.to(device=device)

    # -----------------------------------
    # optimizer
    # -----------------------------------
    lr_rate = 0.01
    milestones = [110000, 150000, 210000]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr_rate, momentum=0.2)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)

    # -----------------------------------
    # data_loader
    # -----------------------------------
    train_img_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/train2017'
    train_label_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/label_train/id_labels'
    val_img_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/val2017'
    val_label_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/label_val/id_labels'

    batch_size = 10
    epochs = 20
    shuffle = True
    train_iter = 117266//batch_size
    val_iter = 4952
    hw = (400, 400)
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        v2.Resize(hw, antialias=True)
    ])
    target_transform = v2.Compose([
        v2.Resize(hw, antialias=True)
    ])
    target_info = (num_classes+1, hw[0], hw[1])

    train_data_loader   = dataloader(train_img_path, train_label_path, target_info, batch_size, 'train', transform, target_transform, shuffle)
    val_data_loader     = dataloader(val_img_path, val_label_path, target_info, 1, 'val', transform, target_transform)

    # -----------------------------------
    # Log
    # -----------------------------------
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H.%M')
    log_dir         = os.path.join('logs', f"{model.backbone.block.__name__}", "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir)

    # -----------------------------------
    # fit one epoch (train & validation)
    # -----------------------------------
    for epoch in range(1, epochs+1):
        fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, train_iter, val_iter, train_data_loader, val_data_loader, loss_history, num_classes+1, save_period=1, device=device)

    loss_history.writer.close()