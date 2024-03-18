import torch

import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, n_observe, n_actions):
        super(NeuralNetwork, self).__init__()
        
        self.fc1 = nn.Linear(n_observe, 512)
        self.fc2 = nn.Linear(512, 512)
        # self.fc3 = nn.Linear(128, n_actions)
        self.fc_final = nn.Linear(512, n_actions)
        
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_final(x)

        return x