import sys
#sys.stdout = LogFile('memory_profile_log', reportIncrementFlag=False)
sys.path.append("../")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms, utils
import cv2
from datareader_toy import toyScenesdata
from scipy.signal import resample

from models.framework import FCNPastProcess
import utils.config as cfg
from memory_profiler import profile

# from nuscenes.map_expansion.map_api import NuScenesMap

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': cfg.BATCH_SIZE,
          'shuffle': cfg.SHUFFLE,
          'drop_last': False,
          'num_workers': cfg.NUM_WORKERS}

# set_name: {'train', 'val', 'test', 'mini_train', 'mini_val', 'train_detect', 'train_track'}
dataset = toyScenesdata()
dataloader = DataLoader(dataset, **params)


def downsample(coords):
    '''
    input shape: [batch_size, 6000, num_agents, coordinate_dimention]
    '''
    # time_interval = 1000
    sample_rate = 500 
    num_bins = int(6000 / sample_rate)
    coords = resample(coords, num_bins, axis=1)
    coords = torch.from_numpy(coords)
    return coords


def main():

    model = FCNPastProcess(32)
    model.double()

    optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)
    nEpochs = 1

    for epoch in range(nEpochs):
        torch.cuda.empty_cache()

        for batch, data in enumerate(dataloader):
            print(batch, len(data))
            print((data["map"]).shape)
            map = np.array(data["map"][0,:])
            #cv2.imshow("map",map)
            #cv2.waitKey(0)
            # break
            coords, gt, = data["coords"], data["ground_truth"]

            #print input data shapes
            print(coords.shape)
            print("=============================")
            print(gt.shape)
            # optimizer.zero_grad()
            
            # coords = downsample(coords)
            # coords = coords[:,:5, :,:]
            # coords_flatten = coords.permute(0,2,1,3)
            # print(coords_flatten.shape)
            # print("-----------------------------------")
            # coords_flatten = coords_flatten.flatten(1,3)
            # print(coords_flatten.shape)
            
            # #num-batches * (num_agents * history)
            # out = model(coords_flatten)

            #TODO: understand what the ground truth is

            # #define loss function
            # criterion = nn.MSELoss()
            # loss = criterion(out, gt)
            # model.zero_grad()
            # loss.backward()
            # optimizer.zero_grad()
            # optimizer.step()
            break


if __name__ == "__main__":
    main()


            

