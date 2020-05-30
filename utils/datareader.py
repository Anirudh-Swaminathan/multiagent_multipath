import sys
sys.path.append('../')
sys.path.append('../nuscenes_devkit/python_sdk/')

from itertools import chain
from typing import List
import numpy as np
import json
import cv2
import os
import csv
from torch.utils.data import Dataset
import imageio
from nuscenes_devkit.python_sdk.nuscenes import *
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.utils.splits import create_splits_scenes

class nuScenesdata(Dataset):

    def __init__(self):

        data_path = "/home/krungta/ECE_285/data/sets/nuscenes/" #path to data stored
        self.nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=True)
#         self.trainset = get_prediction_challenge_split("mini_train", dataroot=data_path)
        self.helper = PredictHelper(self.nusc)
    
        #get all the scenes
        self.scenes = create_splits_scenes()
        #get all the scenes in the trainset
        self.set_name = "train"
        self.trainset = self.scenes[self.set_name]
        self.prediction_scenes = json.load(open(data_path+"maps/prediction_scenes.json", "r"))
        
        print("Number of samples in train set: %d" % (len(self.trainset)))


    def __len__(self):
        return len(self.trainset) #return length of labels or input should be the same

    def __getitem__(self, idx):
        
        #get the scene
        print(self.trainset)
        scene = self.trainset[0]
#         print(scene)
        
        #get all the tokens in the scene
        scene_tokens = self.prediction_scenes[scene]
#         print(scene_tokens)
        
        if len(scene_tokens) < 2:
            print("Not enough agents in the scene")
            return []
        
        #get the tokens in the scene: we will be using the instance tokens as that is the agent in the scene
        tokens = [scene_tok.split("_") for scene_tok in scene_tokens]
        instance_tokens, sample_tokens = tokens[:][0], tokens[:][1]
        
        
#         #extract the sample and get all the data (agents) pertaining to that
#         sample = self.helper.get_annotations_for_sample(sample_token)
        
        #get all the future data for the sample
        future_instance_global = self.helper.get_future_for_agent(instance_token,sample_token, seconds=3, in_agent_frame = False)
        
        #how to get the annotation for the instance in the sample
        annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        
        #get the heading change rate of the agent
        heading_change_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)
        
        return scene
        