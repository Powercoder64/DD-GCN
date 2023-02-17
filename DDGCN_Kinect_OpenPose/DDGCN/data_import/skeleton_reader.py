# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# operation


class SkeletonReader(torch.utils.data.Dataset):

    def __init__(self,
                 data_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):

        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        
        self.data =  np.expand_dims(self.data, axis=0)
        
        if (self.data.shape[2] != 300):
            self.data = temporal_sampling(self.data, 300)  
            
            self.data = np.concatenate((self.data, self.data), axis=0, out=None)

            self.N, self.C, self.T, self.V, self.M = self.data.shape
             
        else:
            
        
            self.data = np.concatenate((self.data, self.data), axis=0, out=None)

            self.N, self.C, self.T, self.V, self.M = self.data.shape
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data_numpy = np.array(self.data[index])



        return data_numpy
    
    
def temporal_sampling(np_array, size, random_crop=False):

    T = np_array.shape[2]

    if T >= size:
        if random_crop:
            np_array = np_array[:, :, random.randint(0, T -
                                                     size):][:, :, :size]
        else:
            np_array = np_array[:, :, :size]

    else:
        selected_index = np.arange(T)
        selected_index = np.concatenate(
            (selected_index, selected_index[1:-1][::-1]))
        selected_index = np.tile(selected_index,
                                 size // (2 * T - 2) + 1)[:size]

        np_array = np_array[:, :, selected_index]


    return np_array
    