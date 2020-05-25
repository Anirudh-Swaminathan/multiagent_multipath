import numpy as np
import cv2
import os
import csv
from torch.utils.data import Dataset
import imageio


data_path = "" #path to data stored

class nuScenesdata(Dataset):

    def __init__(self):

        self.data_dir = "" #path to data lists

        self.input = "" #read all the data that could be inputted into the network
        self.labels = "" #read all the labels that could be inputted into the network


    def __len__(self):
        return len(self.input) #return length of labels or input should be the same

    def __getitem__(self, idx):

        #return an individual input into the network                                                
