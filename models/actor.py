import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super(Actor, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Sigmoid())


    def forward(self, state):
        return self.sequential(state)