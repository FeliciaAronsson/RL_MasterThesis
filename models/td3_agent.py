import torch
import torch.optim as optim
from models.actor import Actor
from models.critic import Critic
from utils.replay_buffer import ReplayBuffer
import numpy as np
import torch.nn.functional as F

class TD3Agent:
    def __init__(self, obs_dim, act_dim, hidden_dim, tau, gamma, learnRate, policy_delay = 2):
        self.tau = tau
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.total_it = 0

        # Actor
        self.actor = Actor(obs_dim, act_dim, hidden_dim)
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.opt_actor = optim.Adam(self.actor.parameters(), lr= learnRate)

        # Twin Critics
        self.critic1 = Critic(obs_dim, act_dim, hidden_dim)
        self.critic2 = Critic(obs_dim, act_dim, hidden_dim)
        self.critic_target1 = Critic(obs_dim, act_dim, hidden_dim)
        self.critic_target2 = Critic(obs_dim, act_dim, hidden_dim)

        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        self.opt_critic = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr = learnRate)

        # Replay buffer
        self.buffer = ReplayBuffer()


    def select(self, s, noice_scale):
        with torch.no_grad():
            action = self.actor(torch.tensor(s).float().unsqueeze(0)).item()

            # Brus
            if noice_scale > 0.0:
                action += np.random.normal(0, noice_scale)
        
            return np.clip(action, 0.0, 1.0)



    def train(self, batch=64):
        self.total_it += 1

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
            target_q1 = r + self.gamma * (1 - d) * self.critic_target1(s2, target_a)
            target_q2 = r + self.gamma * (1 - d) * self.critic_target2(s2, target_a)
            target_q = r + self.gamma * (1 - d) * torch.min(target_q1, target_q2)

        # Critic update
        q_1 = self.critic1(s, a)
        q_2 = self.critic2(s, a)
        critic_loss = F.mse_loss(q_1, target_q) + F.mse_loss(q_2, target_q)
        
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # Delayed policy updates 
        if self.total_it % self.policy_delay == 0:

            # Actor update
            actor_loss = -self.critic1(s, self.actor(s)).mean() 
            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            # Soft updates of target networks
            for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
            for p, pt in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)