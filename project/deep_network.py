# Zach Vincent
# based on code from this online course by Phil Tabor:
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr:float, n_actions:int, input_dims:int, name:str, dir:str):
        '''
        Build a DeepQNetwork object.
        Params:
            lr (float): learning rate
            n_actions (int): number of actions in action space
            input_dims (int): number of inputs to neural net
            name (str): name of checkpoint file
            dir (str): checkpoint file path
        '''
        assert type(input_dims) == int, "[ERROR] DeepQNetwork not given integer input_dims"
        super(DeepQNetwork, self).__init__()
        # save checkpoint files during model training
        self.dir = dir
        self.file = os.path.join(self.dir, name)

        # generate neural network with input from observations
        # hidden layer
        # output n_actions
        self.fc1 = nn.Linear(input_dims, 32)
        self.fc2 = nn.Linear(32, n_actions)
        
        # set optimizer and loss functions
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # since processing isn't parallel, use CPU instead of GPU
        self.device = T.device('cpu')
        T.set_num_threads(2)
        print(f'Using {self.device}')
        self.to(self.device)

    # calculate feed forward
    # takes current state and returns list of action probabilities
    def forward(self, state):
        '''
        Propagates state into NN (feeds forward).

        Params:
            state: current state of the environment.

        Returns: list of action probabilities.
        '''

        hidden = F.relu(self.fc1(state))
        actions = self.fc2(hidden)

        return actions        


    def save_checkpoint(self, net_name:str) -> None:
        '''
        Saves current training progress as a checkpoint.
        Params:
            net_name (str): name of net being saved.
        '''

        print(f'[INFO] Saving checkpoint {net_name}...')
        T.save(self.state_dict(), self.file)


    def load_checkpoint(self, net_name:str) -> None:
        '''
        Loads checkpoint for net testing and evaluation.
        Params:
            net_name (str): name of net being loaded.
        '''

        print(f'[INFO] Loading checkpoint {net_name}...')
        # loads dictionary from file
        self.load_state_dict(T.load(f'models/best_{net_name}'))
