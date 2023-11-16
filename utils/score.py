from typing import Any
import torch
import torch.nn as nn
import numpy as np

def loss_bce(output, target, mode):
    m = nn.Softmax(dim=1)
    output = m(output)
    if mode == 'val':
        device = output.device
        target_t = torch.zeros(1, output.shape[2], output.shape[3], device=device)
        target_t = target_t == target
        for i in range(1, output.shape[1]):
            class_filter = torch.ones(1, output.shape[2], output.shape[3], device=device)*i
            class_filter = class_filter == target
            target_t = torch.cat((target_t, class_filter), 1)
        target = target_t * 1.0
        target.to(device=device)
    loss_fn = nn.BCELoss()
    loss = loss_fn(output, target)

    return loss

def miou(output, target):
    num_classes = output.shape[1]
    m = nn.Softmax(dim=1)
    output = m(output)
    _, output = torch.max(output, 1)

    ious = np.zeros(num_classes)
    for i in range(num_classes):
        f = target != i
        pred_p = output == i
        inter = pred_p.sum()
        union = (target == i).sum() + (f == pred_p).sum()
        ious[i] = (1.0*inter/(union+2.220446049250313e-16)).data.cpu()
    
    miou = ious.mean()
    
    return miou
