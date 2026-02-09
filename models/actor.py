import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(state_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, action_dim), nn.Sigmoid())


    def forward(self, state):
        return self.net(state)