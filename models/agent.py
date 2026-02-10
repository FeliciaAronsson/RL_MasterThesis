
from models.actor import Actor
from models.critic import Critic
import torch
import torch.nn as nn
import torch.optim as optim
from utils.replay_buffer import ReplayBuffer
def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(
            tau * s_param.data + (1.0 - tau) * t_param.data
        )


class DDPGAgent:
    def __init__(self, obs_dim, act_dim, hidden_dim, tau, gamma):
        self.actor = Actor(obs_dim, act_dim, hidden_dim)
        self.critic = Critic(obs_dim, act_dim, hidden_dim)

        self.target_actor = Actor(obs_dim, act_dim, hidden_dim)
        self.target_critic = Critic(obs_dim, act_dim, hidden_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.tau = tau
        self.gamma = gamma
        self.buffer = ReplayBuffer()

    def update_targets(self):
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def select(self, state):
        with torch.no_grad():
            return self.actor(torch.tensor(state).float().unsqueeze(0)).item()

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
            #Bellman
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