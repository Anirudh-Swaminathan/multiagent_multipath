#!/usr/bin/env python

# python2 and python3 compatibility between loaded modules
from __future__ import print_function

import sys
sys.path.append("../")

# All imports here
# Reading files
import os

# Vector manipulations
import numpy as np
import pandas as pd

# DL framework
# torch
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
from torch.nn.utils.rnn import pack_padded_sequence

# import toy dataset class
from utils.datareader_toy import toyScenesdata

# Plotting images
from matplotlib import pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# PIL Image
from PIL import Image

# regex for captions
import re

# import nntools
import models.nntools_modified as nt

# import add for fast addition between lists
from operator import add

# json for dumping stuff onto files as output
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

import utils.config as cfg

# TODO - remove this unnecessary class #Done
# class for the dataset
# class ToyDataset(td.Dataset):
#     """ Class to hold the Toy Dataset """

#     def __init__(self, root_dir, mode="train", image_size=(500, 500)):
#         super(ToyDataset, self).__init__()
#         self.image_size = image_size
#         self.mode = mode

#     def __len__(self):
#         pass

#     def __repr__(self):
#         return "ToyDataset(mode={}, image_size={})".format(self.mode, self.image_size)

#     def __getitem__(self):
#         pass



# NN Classifier from nntools
class NNClassifier(nt.NeuralNetwork):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def criterion(self, y, d):
        return self.cross_entropy(y, d)



class CNNSceneContext(nn.Module):
    def __init__(self, num_out, fine_tuning=True):
        super(CNNSceneContext, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = fine_tuning
        self.features = vgg.features
        # the average pooling is the same
        self.avgpool = vgg.avgpool
        # the classifier is also the same
        self.classifier = vgg.classifier
        # CODE to change the final classifier layer
        num_ftrs = vgg.classifier[6].in_features
        self.classifier[6] = nn.Linear(num_ftrs, num_out)

    def forward(self, x):
        # COMPLETE the forward prop
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)
        f = self.classifier(f)
        return f



# class RNNAnchorProcess(nn.Module):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         pass


# class RNNPastProcess(nn.Module):
#     def __init__(self):
#         pass

#     def forward(self):
#         pass


#     def greedy_sample(self):
#         """ Method to greedily sample from the RNN """
#         pass

class FCNPastProcess(nn.Module):
    def __init__(self, fdim):
        '''
        fdim: Num channels in the output (number of intents)
        '''

        super(FCNPastProcess, self).__init__()
        self.fc1 = nn.Linear(cfg.PAST_TRAJECTORY_LENGTH * cfg.NUM_AGENTS * 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, fdim)
        

    def forward(self, x):
        '''
        x dim: PAST_TRAJECTORY_LENGTH * NUM_AGENTS * BATCH_SIZE
        '''
        # assert (x.size() == torch.Size([2,PAST_TRAJECTORY_LENGTH])), print("Incorrect tensor shape passed to FCN")
        print("In model: ", x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)

        
class IntentionEmbedding(nn.Module):
    def __init__(self, fdim, out_dim):
        # fdim: dim of past trajectory features
        super(IntentionEmbedding, self).__init__()
        self.emb=nn.Sequential(
            nn.Linear(4, 32)
        )
        self.encode=nn.Sequential(
            nn.Linear(32+fdim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )
        
        
    def forward(self, x, intention):
        # x: batch x n_vehicles x fdim, past trajectory features
        # intention: batch x n_vehicles x 4, one-hot embedding of intentions
        intention=self.emb(intention)
        y=torch.cat((x, intention), dim=1)
        y=self.encode(y)
        return y


class ScoringFunction(nn.Module):
    def __init__(self, fdim):
        super(ScoringFunction, self).__init__()
        self.score=nn.Sequential(
            nn.Linear(fdim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        #self.sm=nn.Softmax()

    def forward(self, x):
        # x: batch x fdim (scene+past_intent_embedding)
        y=self.score(x).squeeze()
        #y=self.sm(y)
        return y
        
class MultiAgentNetwork(NNClassifier):
    def __init__(self, n_intents, scene_out, intent_in, intent_out, score_in, fine_tuning=True):
        """
        n_intents - number of intents in dataset
        scene_out - dimension of the output for the scene parsing CNN
        intent_in - dimension of the input for the intention embedding
        intent_out - dimension of the output for the intention embedding
        score_in - dimension of the input of the scoring module
        """
        super(MultiAgentNetwork, self).__init__()
        self.scene = CNNSceneContext(scene_out, fine_tuning)  
        self.past = FCNPastProcess() 
        self.intent = IntentionEmbedding(intent_in, intent_out)
        self.score = ScoringFunction(scene_out+intent_out)
        self.n_intents = n_intents
        

    def forward(self, img, past_traj, gt_future):
        scene_output = self.scene(img)
        # TODO - check if shape[0] and shape[1] are correct
        n_agents = past_traj.shape[0]
        n_modes = self.n_intents**n_agents
        scores = torch.zeros(self.n_modes)
        # TODO: compute past_output with self.FCNPastProcess
        # TODO: compute ground truth tensor. (sum_{agent} (agent**self.n_intents)*agent_intention)
        for mode in range(n_modes):
            intentions = torch.zeros(past_output.shape[0], n_agents, self.n_intents)
            for agent in n_agents:
                intention_index = int(mode/self.n_intents**(agent))%self.n_intents
                intentions[..., agent, intention_index] = 1
            # past_output: (n_batch, n_vehicles, fdim)
            traj_output = self.intent(past_output, intentions)
            # traj_output: (n_batch, n_vehicles, intent_out)
            traj_output = F.sum(traj_output, dim=1).squeeze() # or mean, or max
            # traj_output: (n_batch, intent_out)
            combined_output = torch.cat((scene_output, traj_output), dim=1)
            # combined_output: (nbatch, scene_out+intent_out)
            scores[mode] = self.score(combined_output)
        #scores = F.softmax(scores)
        return scores



class ToyStatsManager(nt.StatsManager):
    def __init__(self):
        super(ToyStatsManager, self).__init__()

    def init(self):
        super(ToyStatsManager, self).init()
        self.running_accuracy = 0


    def accumulate(self, loss, x, y, d):
        # TODO - modify the input params to accept the past trajectories also
        super(ToyStatsManager, self).accumulate(loss, x, y, d)
        
        # get the indices of the maximum activation of softmax for each sample
        _, l = torch.max(y, 1)

        # count the running average fraction of correctly classified samples
        self.running_accuracy += torch.mean((l == d).float())

    def summarize(self):
        # this is the average loss when called
        loss = super(ToyStatsManager, self).summarize()
        
        # this is the average accuracy percentage when called
        accuracy = 100 * self.running_accuracy / self.number_update
        return {'loss' : loss, 'accuracy' : accuracy}




# class to house training stuff
class TrainNetwork(object):
    def __init__(self):
        self._init_paths()
        #TODO - change the options for the toy dataset, including batch size #DONE
        params = {'batch_size': cfg.BATCH_SIZE,
          'shuffle': cfg.SHUFFLE,
          'drop_last': cfg.DROP_LAST,
          'num_workers': cfg.NUM_WORKERS}

        self.training_dataset = toyScenesdata()
        # self.train_loader = td.Dataloader(self.training_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, pin_memory=True)
        self.train_loader = td.Dataloader(self.training_dataset, **params)
        self.val_dataset = toyScenesdata(set_name="val")
        self.val_loader = td.Dataloader(self.val_dataset, batch_size=cfg.BATCH_SIZE, pin_memory=True)
        self._init_train_stuff()

    def _init_paths(self):
        # data loading
        #TODO - change directories
        self.dataset_root_dir = cfg.DATA_PATH

        # output directory for training checkpoints
        # This changes for every experiment
        # self.op_dir = "../nntools_modifiedoutputs/" + <op_dir>
        self.op_dir = cfg.OUTPUT_PATH #+ <experiment nunmber>

    def _init_train_stuff(self):
        self.lr = 1e-3
        # TODO Change these values
        self.n_intents = 4
        self.scene_out = 32 #fdim
        # self.intent_in = <>
        # self.intent_out = <>
        # self.score_out = <>
        net = MultiAgentNetwork(self.n_intents, self.scene_out, self.intent_in, self.intent_out, self.score_out)
        self.net = net.to(device)
        self.adam = torch.optim.Adam(net.parameters(), lr=self.lr)
        self.stats_manager = ToyStatsManager()
        # TODO - change the output_dir #DONE
        self.exp = nt.Experiment(self.net, self.training_dataset, self.val_dataset, self.adam, self.stats_manager, output_dir=self.op_dir, perform_validation_during_training=True)

 
    def myimshow(self, img, ax=plt):
        image = image.to('cpu').numpy()
        image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
        image = (image + 1) / 2
        image[image<0] = 0
        image[image>1] = 1
        h = ax.imshow(image)
        ax.axis('off')
        return h


    def plot(self, exp, fig, axes):
        axes[0].clear()
        axes[1].clear()
        # Plot the training loss over the epochs
        axes[0].plot([exp.history[k][0]['loss'] for k in range(exp.epoch)], label="training loss")
        # Plot the evaluation loss over the epochs
        axes[0].plot([exp.history[k][1]['loss'] for k in range(exp.epoch)], color='orange', label="evaluation loss")
        # legend for the plot
        axes[0].legend()
        # xlabel and ylabel
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        # Plot the training accuracy over the epochs
        axes[1].plot([exp.history[k][0]['accuracy'] for k in range(exp.epoch)], label="training accuracy")
        # Plot the evaluation accuracy over the epochs
        axes[1].plot([exp.history[k][1]['accuracy'] for k in range(exp.epoch)], label="evaluation accuracy")
        # legend for the plot
        axes[1].legend()
        # xlabel and ylabel
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        plt.tight_layout()
        # set the title for the figure
        # fig.suptitle("Loss and Accuracy metrics")
        fig.canvas.draw()

    def run_plot_exp(self):
        fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
        self.exp.run(num_epochs=20, plot=lambda exp: self.plot(exp, fig=fig, axes=axes))

    def run_exp(self):
        # RUN on the server, without plotting
        self.exp.run(num_epochs=20)

    def save_evaluation(self):
        exp_val = self.exp.evaluate()
        with open(self.op_dir+'val_result.txt', 'a') as t_file:
            print(exp_val, t_file)


def main():
    tn = TrainNetwork()
    tn.run_exp()
    tn.save_evaluation()


if __name__ == '__main__':
    main()
