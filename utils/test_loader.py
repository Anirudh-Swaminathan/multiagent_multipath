import os
import torch

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from datareader_toy import toyScenesdata
# from nuscenes.map_expansion.map_api import NuScenesMap

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'drop_last': False,
          'num_workers': 1}

# set_name: {'train', 'val', 'test', 'mini_train', 'mini_val', 'train_detect', 'train_track'}
dataset = toyScenesdata()
dataloader = DataLoader(dataset, **params)

for batch, data in enumerate(dataloader):
    print(batch, len(data))
    print((data["map"]).shape)
    map = np.array(data["map"][0,:])
    cv2.imshow("map",map)
    cv2.waitKey(0)
    break
