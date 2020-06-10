import sys
sys.path.append("../")

import shutil
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import resample

def downsampling(coords):
    '''
    input shape: [batch_size, 6000, num_agents, coordinate_dimention]
    '''
    # time_interval = 1000
    sample_rate = 500 
    num_bins = int(6000 / sample_rate)
    coords = coords[::500]
    return coords

ORIGINAL_DATA_PATH = "../data/toydataset/"
NEW_DATA_PATH = "../data/toydataset_resampled/"

if __name__ == "__main__":    

    scenes = os.listdir(ORIGINAL_DATA_PATH)
    
    if not os.path.isdir(NEW_DATA_PATH):
            os.makedirs(NEW_DATA_PATH)
            print("Creating folder for dataset; ")

    for scene in scenes:

        print("========================= working on scene", scene, "=========================")
        cur_path = NEW_DATA_PATH+scene
        if not os.path.isdir(cur_path):
            os.makedirs(cur_path)
            print("Creating folder for scene; ", scene)

        print("Copying image from original to new without any alterations")

        if os.path.exists(cur_path+"/scene.png"):
            print("Image exists. Don't need to copy")
        else:
            shutil.copy(ORIGINAL_DATA_PATH+scene+"/scene.png",cur_path+"/scene.png")

        print("Copying history from original to new after downsampling")

        if os.path.exists(cur_path+"/ls.npy"):
            print("Coords exists. Don't need to copy")
        else:
            coords = np.load(ORIGINAL_DATA_PATH+scene+"/ls.npy", allow_pickle=True)
            print("Original shape")            
            print(coords.shape)

            print("Downsampled shape ")
            resampled = downsampling(coords)
            print(resampled.shape)

            resampled = resampled.transpose(1,0,2)
            print(resampled.shape)        
            print("-----------------------------------")
            coords_flatten = np.resize(resampled, (2,24))
            print(coords_flatten.shape)
            np.save(cur_path+"/ls.npy", coords_flatten, allow_pickle=True)

        if os.path.exists(cur_path+"/init.npy"):
            print("GT exists. Don't need to copy")
        else:
            coords = np.load(ORIGINAL_DATA_PATH+scene+"/init.npy", allow_pickle=True)
            print("Original shape ")            
            print(coords.shape)

            print("Intentions")
            vals = coords[:,2]
            print(vals)
            print(vals.shape)

            np.save(cur_path+"/init.npy", vals, allow_pickle=True)

        print("Finished processing scene")
        
    print("Finished processing all")






    # for batch, data in enumerate(dataloader):
    #     # print(batch, len(data))
    #     # print((data["map"]).shape)
    #     # map = np.array(data["map"][0,:])
    #     # cv2.imshow("map",map)
    #     # cv2.waitKey(0)
    #     # break

    #     coords, gt, = data["coords"], data["ground_truth"]

    #     #print input data shapes
    #     print(coords.shape)
    #     print("=============================", batch, "=============================")
    #     print(gt.shape)
        
    #     coords = downsample(coords)
    #     coords = coords[:,:5, :,:]
    #     coords_flatten = coords.permute(0,2,1,3)
    #     