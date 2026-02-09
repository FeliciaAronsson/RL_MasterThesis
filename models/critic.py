import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1)
        )


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

