from typing import Any
import torch
import torch.nn as nn
import numpy as np

def loss_bce(output, target, mode):
    m = nn.Softmax(dim=1)
    output = m(output)
    device = output.device
    if mode == 'val':
        target_t = torch.zeros(1, output.shape[2], output.shape[3], device=device)
        target_t = target_t == target
        for i in range(1, output.shape[1]):
            class_filter = torch.ones(1, output.shape[2], output.shape[3], device=device)*i
            class_filter = class_filter == target
            target_t = torch.cat((target_t, class_filter), 1)
        target = target_t * 1.0
        target.to(device=device)
    loss_weight = torch.ones_like(output, device=device) * 10
    loss_weight[0, 0] = loss_weight[0, 0] * (1/10)
    loss_fn = nn.BCELoss(loss_weight)
    loss = loss_fn(output, target)

    return loss

def miou(output, target, total_iters=0, total_unions=0):
    num_classes = output.shape[1]
    m = nn.Softmax(dim=1)
    output = m(output)
    _, output = torch.max(output, 1)
    output = output.unsqueeze(1)
    for i in range(num_classes):
        f = target != i
        p = target == i
        pred_p = output == i
        total_iters[i] += (pred_p * p).sum()
        total_unions[i] += p.sum() + (f * pred_p).sum()
    
    return total_iters, total_unions
