import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from datareader import nuScenesdata
from nuscenes.map_expansion.map_api import NuScenesMap

dataset = nuScenesdata()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=1)

for batch, data in enumerate(dataloader):
    print(batch, (data))
    break
