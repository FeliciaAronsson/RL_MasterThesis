import torch.nn as nn 

class QNetwork(nn.Module):
    """
    Q-network for DQN agent, estimates Q-values for each action given a state. 
    """

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim) # Q(s,a) 
        )

    def forward(self, s):
        return self.net(s)