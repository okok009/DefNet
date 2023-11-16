import torch
import torch.nn as nn
import numpy as np


# a = torch.rand((1, 3, 3, 3)) * 3
# m = nn.Softmax()
# a = m(a)
# print(a)
# _, a = torch.max(a, 1)
# print(a, _)
# b = torch.ones((1, 1, 3, 3))
# ious = np.zeros(91)
# for i in range(3):
#     inter = (a == i).sum()
#     c = b!=i
#     d = a==i
#     union = (b == i).sum() + (c==d).sum()
#     ious[i] = (1.0*inter/(union+2.220446049250313e-16)).data.cpu()


# print(ious)
# print(ious.mean())
a = torch.zeros((1, 3, 3))
b = torch.zeros((1, 3, 3))
c = torch.cat((a, b), 0)
print(c.shape)
d = torch.cat((c, a), 0)
print(d.shape)
for i in range(1, 10):
    print(i)