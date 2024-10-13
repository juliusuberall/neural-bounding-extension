import torch.nn as nn
import torch


class OursNeural2D(nn.Module):
    def __init__(self, input_dim):
        super(OursNeural2D, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.sigmoid(self.fc3(x2))
        
        x2 = torch.sigmoid(self.fc2(x1))
        x1 = torch.sigmoid(self.fc1(x))
        
        return x1, x2, x3
