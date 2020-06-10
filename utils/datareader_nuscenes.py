import sys

#comment if you used pip install else append the appropriate paths
# sys.path.append('../')
# sys.path.append('../nuscenes_devkit/python_sdk/')

from collections import Counter
from collections import OrderedDict
from itertools import chain
from typing import List
import numpy as np
import pandas as pd
import json
import cv2
import os
import csv
from torch.utils.data import Dataset
import imageio
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.utils.splits import create_splits_scenes


import matplotlib.pyplot as plt
# %matplotlib inline
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

DATA_PATH = "../data/sets/nuscenes/" #path to data stored
DATA_VERSION = 'v1.0-mini'
NUM_AGENTS = 2
TRAJECTORY_TIME_INTERVAL = 6 #length (in time) per data point


class nuScenesdata(Dataset):

    def __init__(self, set_name="mini_train"):
        
        #assert statements
        set_paths = ['train', 'val', 'test', 'mini_train', 'mini_val', 'train_detect', 'train_track']
        assert set_name in set_paths, "Incorrect set_name"
        
        #Initialize data and Prediction Helper classes
        self.data_path = DATA_PATH
        self.nusc = NuScenes(version=DATA_VERSION, dataroot=self.data_path, verbose=True)
        self.helper = PredictHelper(self.nusc)
    
        #get all the scenes
        self.scenes = create_splits_scenes()
        
        #get all the scenes in the trainset
        self.set_name = set_name
        self.trainset = self.scenes[self.set_name] #List of scenes as part of training set
        self.prediction_scenes = json.load(open(self.data_path+"maps/prediction_scenes.json", "r")) #Dictionary containing list of instance and sample tokens for each scene
        
        print("Number of samples in train set: %d" % (len(self.trainset)))

    def __len__(self):
        return len(self.trainset) #return length of labels or input should be the same

    def __getitem__(self, test_idx):
        
        #get the scene
        scene = self.trainset[test_idx]
        
        #get all the tokens in the scene
        #List of scene tokens in the given scene where each item comprises of an instance token and a sample token seperated by underscore
        scene_tokens = self.prediction_scenes[scene] 
      
        #Return if fewer than 2 tokens in this scene
        if len(scene_tokens) < 2:
            print("Not enough agents in the scene")
            return []
        
        #get the tokens in the scene: we will be using the instance tokens as that is the agent in the scene
        tokens = [scene_tok.split("_") for scene_tok in scene_tokens]
        
        #List of instance tokens and sample tokens
        instance_tokens, sample_tokens = list(list(zip(*tokens))[0]), list(list(zip(*tokens))[1]) 
        
        assert len(instance_tokens) == len(sample_tokens), "Instance and Sample tokens count does not match"
        
        
        '''
        1. Convert list of sample and instance tokens into an ordered dict where sample tokens are the keys
        2. Iterate over all combinations (of length TRAJECOTRY_TIME_INTERVAL) of consecutive samples 
        3. Form a list of data points where each data point has TRAJECOTRY_TIME_INTERVAL sample tokens where 
            each sample token has data for all instance tokens identified in step 2
        4. Create 3 numy arrays each for coordinates, heading_change_rate and map with appropriate shapes
        5. Iterate: per sample per instance and fill in numpy arrays with respective data
        6. Form a dict containing the 3 numpyarrays and return it
        '''
        
        
        
        ordered_tokens = OrderedDict(zip(sample_tokens, instance_tokens))
        
        print("Printing Ordered_tokens: ", ordered_tokens)
        return []
        
        
        
        
        
        
        
        
        
        
        
        
        
        #Dictionary containing count for number of samples per token
        token_count = Counter(instance_tokens) 
        
        #used to find n agents with highest number of sample_tokens
        minCount = sorted(list(token_count.values()), reverse=True)[NUM_AGENTS-1] 
    
        #Convert isntance and sample tokens to dict format
        instance_sample_tokens = {}
        for instance_token, sample_token in zip(instance_tokens, sample_tokens):
            if token_count[instance_token] >= minCount:
                try:
                    instance_sample_tokens[instance_token].append(sample_token)
                except:
                    instance_sample_tokens[instance_token] = [sample_token]
                
#         print("Instance:samples ===============================================================================")
#         print(instance_sample_tokens)

        if len(list(instance_sample_tokens.keys())) != NUM_AGENTS:
            print()
#             print("Instance_sample_tokens: \n", instance_sample_tokens)         
        '''
        Format: 
        {coordinates: [[coord_at_t0, coord_at_t1, coord_at_t2, ..., coord_at_tTAJECTORY_TIME_INTERVAL],...numDatapointsInScene ], 
         heading_change_rate; [[h_at_t0, h_at_t1, h_at_t2, ..., h_at_tTAJECTORY_TIME_INTERVAL], ...numDatapointaInScene] 
        }
        '''
        
        #Initialize map rasterizers
        static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=2.5)
        mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())
        
        #Initialize Output data
        output_data = {"coordinates": np.zeros((len(instance_sample_tokens.keys()), 1)), "heading_change_rate": np.zeros((len(instance_sample_tokens.keys()), 1)), "map":[0]*len(instance_sample_tokens.keys())}
        
        for t,instance_token in enumerate(instance_sample_tokens.keys()):

            instance_coordinates = np.zeros((int(len(instance_sample_tokens[instance_token]) / TRAJECTORY_TIME_INTERVAL),TRAJECTORY_TIME_INTERVAL,3))
            instance_heading_change_rate = np.zeros((int(len(instance_sample_tokens[instance_token]) / TRAJECTORY_TIME_INTERVAL),TRAJECTORY_TIME_INTERVAL))
            
            print("Shape of instance_coordinates: ", instance_coordinates.shape)
            idx = 0                       #0 --> numData points for this instance (dimension 1)
            num = 0                       #0 --> TRAJECTORY_TIME_INTERVAL (dimension 2)
            for sample_token in (instance_sample_tokens[instance_token]):
#                 print(idx, "     ", num)
#                 print(self.nusc.get('sample', sample_token)["timestamp"]) 

                #how to get the annotation for the instance in the sample
                annotation = self.helper.get_sample_annotation(instance_token, sample_token)
                instance_coordinates[idx][num] = annotation["translation"]

                #get the heading change rate of the agent
                heading_change_rate = self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)
                instance_heading_change_rate[idx][num] = heading_change_rate
                
                
                num = num + 1
                
                #reached the number of records per sample
                if num == TRAJECTORY_TIME_INTERVAL:
                    idx = idx+1
                    num = 0
                    
                if idx == instance_coordinates.shape[0]:
                    break
                    
                img = mtp_input_representation.make_input_representation(instance_token, sample_token)
#                 cv2.imshow("map",img)

            output_data["map"][t]=(img)
#             plt.imsave('test'+str(test_idx)+str(t)+'.jpg',img)
            output_data["coordinates"][t]=instance_coordinates
            output_data["heading_change_rate"][t]=instance_heading_change_rate

#         test = pd.DataFrame(output_data,columns=["coordinates", "heading_change_rate", "map"])
#         test.to_csv('test'+str(test_idx)+'.csv')

        print("Printing Output data")
        print((output_data["coordinates"]))
        print(len(output_data["heading_change_rate"]))
        print(len(output_data["coordinates"]))
        
        return output_data
        