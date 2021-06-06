import torch
import numpy as np

hr = torch.load('../dataset_root/DCBs/HR/000000022158.pth.tar')
arr = hr.numpy()
index = 0
for cat in arr:
    if np.any(cat):
        print('Dimension: ' + str(index) + ' distinta de cero')
    index += 1

breakpoint()