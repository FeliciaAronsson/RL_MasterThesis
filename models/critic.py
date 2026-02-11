import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super(Critic, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.sequential(x)