 
import torch
import torch.optim as optim
from models.actor import Actor
from models.critic import Critic
from utils.replay_buffer import ReplayBuffer
import numpy as np

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, hidden_dim, tau, gamma, learnRate):
        self.tau = tau
        self.gamma = gamma

        self.actor = Actor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, act_dim, hidden_dim)

        # Target used for convergens
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dim)

        # Store transition
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr= learnRate)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr= learnRate)

        self.buffer = ReplayBuffer()

    def select(self, state, noice_scale):
        # no_grad - does not update the gradient when updating the weights in the neural network. 
        with torch.no_grad():
            action = self.actor(torch.tensor(state).float().unsqueeze(0)).item()

            # Add noice
            if noice_scale > 0.0:
                action += np.random.normal(0, noice_scale)
        
            # clip make sure that the action is between 0 and 1
            return np.clip(action, 0.0, 1.0)