
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)



from models.actor import Actor
from models.critic import Critic
import torch
import torch.nn as nn
import torch.optim as optim

def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(
            tau * s_param.data + (1.0 - tau) * t_param.data
        )


class DDPGAgent:
    def __init__(self, obs_dim, act_dim, tau, gamma):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim)

        self.target_actor = Actor(obs_dim, act_dim)
        self.target_critic = Critic(obs_dim, act_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.tau = tau
        self.gamma = gamma

    def update_targets(self):
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)


import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size=1_000_000):
        self.buffer = []
        self.size = size


    def add(self, s, a, r, s2, d):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s2, d))


    def sample(self, batch):
        samples = random.sample(self.buffer, batch)
        return map(np.array, zip(*samples))