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
from nets.model import unt_defnet18, unt_defnet50
from data.dataset import SegDataset


# -----------------------------------
# device
# -----------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# -----------------------------------
# model
# -----------------------------------
model = unt_defnet18()

model = model.to(device=device)

# -----------------------------------
# optimizer
# -----------------------------------
lr_rate = 0.01
milestones = [100000, 140000, 160000]
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=lr_rate, momentum=0.2)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)

# -----------------------------------
# data_loader
# -----------------------------------
train_img_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/train2017'
train_label_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/label_train/class_labels'
val_info_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/val2017'
val_label_path = 'E:/ray_workspace/fasterrcnn_desnet/COCODevKit/label_val/class_labels'

batch_size = 5
epochs = 10
shuffle = True
train_iter = 117266//batch_size
val_iter = 1000

train_data_loader = SegDataset(train_img_path, train_label_path)
val_data_loader = SegDataset(val_label_path, val_info_path)

# -----------------------------------
# Log
# -----------------------------------
time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
log_dir         = os.path.join('logs', "loss_" + str(time_str))
loss_history    = LossHistory(log_dir)

# -----------------------------------
# fit one epoch (train & validation)
# -----------------------------------
# training_loop(epochs, optimizer, model, lr_scheduler, iter, train_data_loader)
for epoch in range(1, epochs+1):
    fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, train_iter, val_iter, train_data_loader, val_data_loader, loss_history)

loss_history.writer.close()