# Zach Vincent
# based on code from this online course:
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/
# This code is not functional, it is representative. 

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        # used when deriving one class from another
        super(LinearClassifier, self).__init__()

        # n input dimensions to 128 output dimensions
        self.fc1 = nn.Linear(*input_dims, 128)
        # 128 input, 256 output
        self.fc2 = nn.Linear(128, 256)
        # 256 input, n_classes output
        self.fc3 = nn.Linear(256, n_classes)

                                   # what to optimize  rate of optimization
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # loss function
        self.loss = nn.CrossEntropyLoss()
        # detect GPU
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        # pushes data through activation layers
        # uses sigmoid activation function
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        # loss function activates last layer for us
        layer3 = self.fc3(layer2)

        return layer3

    def learn(self, data, labels):
        # zero out gradients for optimizer in every learning function
        self.optimizer.zero_grad()
        # convert data and labels to tensors
        # lowercase tensor preserves datatype
        data    = T.tensor(data).to(self.device)
        labels  = T.tensor(labels).to(self.device)
        
        # get predictions
        predictions = self.forward(data)

        # determine accuracy of labels with known data
        cost = self.loss(predictions, labels)

        # back propagate cost (actually changes net data)
        cost.backward()
        self.optimizer.step()
