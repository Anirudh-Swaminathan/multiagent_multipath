#!/usr/bin/env python

# python2 and python3 compatibility between loaded modules
from __future__ import print_function

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
import nntools as nt

# import add for fast addition between lists
from operator import add

# json for dumping stuff onto files as output
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



# class for the dataset
class ToyDataset(td.Dataset):
    """ Class to hold the Toy Dataset """

    def __init__(self, root_dir, mode="train", image_size=(500, 500)):
        super(ToyDataset, self).__init__()
        self.image_size = image_size
        self.mode = mode

    def __len__(self):
        pass

    def __repr__(self):
        return "ToyDataset(mode={}, image_size={})".format(self.mode, self.image_size)

    def __getitem__(self):
        pass



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



class RNNAnchorProcess(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class RNNPastProcess(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


    def greedy_sample(self):
        """ Method to greedily sample from the RNN """
        pass


class ScoringFunction(NNClassifier):
    def __init__(self):
        pass

    def forward(self, x):
        pass



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
        #TODO - change the options for the toy dataset, including batch size
        self.training_dataset = ToyDataset(root_dir=self.dataset_root_dir)
        self.train_loader = td.Dataloader(self.training_dataset, batch_size=<batch_size>, shuffle=True, piin_memory=True)
        self.val_dataset = ToyDataset(root_dir=self.dataset_root_dir, mode="val")
        self.val_loader = td.Dataloader(self.val_dataset, batch_size=<batch_size>, pin_memory=True)
        self._init_train_stuff()

    def _init_paths(self):
        # data loading
        #TODO - change directories
	self.dataset_root_dir = <dset_root>
	self.train_dir = <dset_train>
	self.val_dir = <dset_val> 
	self.test_dir = <dset_test>

	# output directory for training checkpoints
	# This changes for every experiment
	self.op_dir = "../outputs/" + <op_dir>

    def _init_train_stuff(self):
        self.lr = 1e-3
        net = ScoringFunction()
        self.net = net.to(device)
        self.adam = torch.optim.Adam(net.parameters(), lr=self.lr)
        self.stats_manager = ToyStatsManager()
        # TODO - change the output_dir
        self.exp = nt.Experiment(self.net, self.training_dataset, self.val_dataset, self.adam, self.stats_manager, output_dir=<output_dir>, perform_validation_during_training=True)

 
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