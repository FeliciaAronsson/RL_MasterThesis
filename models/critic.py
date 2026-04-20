import torch
import torch.nn as nn


class Critic(nn.Module):
    """
    The Critic network for the DDPG and TD3 algorithms. 
    It takes the state and action as input and outputs a single Q-value.
    The architecture consists of two hidden layers with ReLU activations and a final output layer that outputs the Q-value.
    """
    
    def __init__(self, state_dim, action_dim, hidden):
        super(Critic, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))


    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.sequential(x)

