 
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

    def select(self, s, noise_scale):
        with torch.no_grad():
            action = self.actor(torch.tensor(state).float().unsqueeze(0)).item()

            # Noise
            if noise_scale > 0.0:
                action += np.random.normal(0, noise_scale)

            return np.clip(action, 0.0, 1.0)


    def train(self, batch=64):
        if len(self.buffer.buffer) < batch:
            return
 
        s, a, r, s2, d = self.buffer.sample(batch)
        s = torch.tensor(s).float()
        a = torch.tensor(a).float().unsqueeze(1)
        r = torch.tensor(r).float().unsqueeze(1)
        s2 = torch.tensor(s2).float()
        d = torch.tensor(d).float().unsqueeze(1)
 
        # Critic target
        with torch.no_grad():
            target_a = self.actor_target(s2)
            # Bellman
            target_q = r + self.gamma * (1 - d) * self.critic_target(s2, target_a)
 
        # Critic update
        q = self.critic(s, a)
        critic_loss = ((q - target_q) ** 2).mean()
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()
 
        # Actor update
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()
 
        # Soft updates of target networks
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)