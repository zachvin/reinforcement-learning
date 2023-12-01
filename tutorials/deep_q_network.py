import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        # save checkpoint files during model training
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # build convolutional layers to take image as input
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # build input to neural network based on conv layers
        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        # generate neural network with input from conv layers
        # output n_actions
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        # set optimizer and loss functions
        self.optimizer = optim.RMSProp(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # calculate how many output dimensions there are given input_dims
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)

        # 3 convolutional layers
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(np.prod(dims.size()))

    # calculate feed forward
    # takes current state and returns list of action probabilities
    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))

        # conv3 shape = batch size * n_filters * H * W
        # first fully-connected layer takes shape:
        # batch size * # input features
        # view() similar to np.reshape()
        conv_state = conv3.view(conv3.size()[0], -1)
                                #batch size      flatten rest of dims
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        # loads dictionary from file
        self.load_state_dict(T.load(self.checkpoint_file))