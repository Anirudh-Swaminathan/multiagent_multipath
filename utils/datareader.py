import sys
sys.path.append('../')
sys.path.append('../nuscenes_devkit/python_sdk/')

import numpy as np
import cv2
import os
import csv
from torch.utils.data import Dataset
import imageio
from nuscenes_devkit.python_sdk.nuscenes import *
from nuscenes.map_expansion.map_api import NuScenesMap


class nuScenesdata(Dataset):

    def __init__(self):

        data_path = "/home/krungta/ECE_285/data/sets/nuscenes/" #path to data stored
        self.nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=True)
#         nusc_map = NuScenesMap(dataroot=data_path, map_name='singapore-onenorth')


    def __len__(self):
        return 3 #return length of labels or input should be the same

    def __getitem__(self, idx):

        #return an individual input into the network                                                
        sample_token = self.nusc.sample[idx]['token']
        return self.nusc.get('sample', sample_token)
        