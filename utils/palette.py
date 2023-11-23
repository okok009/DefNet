import torch
import torch.nn as nn
import numpy as np



def palette(output):
    device=output.device
    num_classes = output.shape[1]
    m = nn.Softmax(dim=1)
    output = m(output)
    _, indx = torch.max(output, 1)
    output = output==_
    
    i, j, k = torch.meshgrid(torch.linspace(0, 255, steps=int(np.ceil(num_classes**(1/3))), device=device),
                             torch.linspace(0, 255, steps=int(np.ceil(num_classes**(1/3))), device=device),
                             torch.linspace(0, 255, steps=int(np.ceil(num_classes**(1/3))), device=device))
    palette = torch.stack((i, j, k), 0)
    palette = palette.view(3, -1).permute(1, 0)
    pred = torch.stack((output[0, 0] * palette[0, 0], output[0, 0] * palette[0, 1], output[0, 0] * palette[0, 2]), 0)

    for i in range(1, num_classes):
        pred += torch.stack((output[0, i] * palette[i, 0], output[0, i] * palette[i, 1], output[0, i] * palette[i, 2]), 0)
    
    indxs = torch.zeros((pred.shape[1:]), device=device)
    pred = pred.permute(1, 2, 0)
    for i in range(num_classes):
        indx = pred == palette[i]
        indx = indx.permute(2, 0, 1)
        indx = indx[0] * indx[1] * indx[2]
        indxs += indx * i
    pred = pred.cpu().detach().numpy()

    del palette
    return pred, indxs