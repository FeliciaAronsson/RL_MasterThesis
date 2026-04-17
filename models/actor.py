import torch.nn as nn

class Actor(nn.Module):
    """
    The Actor network for the DDPG and TD3 algorithms. 
    It takes the state as input and outputs an action in the range [0, 1]. 
    The architecture consists of two hidden layers with ReLU activations and a final output layer with a Sigmoid activation to ensure the output is between 0 and 1.
    """
    
    def __init__(self, state_dim, action_dim, hidden):
        super(Actor, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Sigmoid())


    def forward(self, state):
        return self.sequential(state)