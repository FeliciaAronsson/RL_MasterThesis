import torch
import torch.optim as optim
from models.actor import Actor
from models.critic import Critic
from utils.replay_buffer import ReplayBuffer
import numpy as np
import torch.nn.functional as F
from utils.ou_noice import OUNoice

class TD3Agent:
    def __init__(self, obs_dim, act_dim, hidden_dim, tau, gamma, learnRate, 
                 policy_noise = 0.2, noise_clip = 0.5, policy_delay = 2):
        self.tau = tau
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
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
        self.noise = OUNoice(mu = np.zeros(act_dim))

        # Replay buffer
        self.buffer = ReplayBuffer()

    def select_ou(self, state, train = True):

            self.actor.eval()

            with torch.no_grad():
                action = self.actor(torch.tensor(state).float().unsqueeze(0)).item()

            self.actor.train()
            
            if train:    #nu är noise här istället för i train ddpg
                action = np.clip(action + self.noise()[0], 0.0, 1.0)

            return action
        
    def reset_noise(self):
        self.noise.reset()


    def select(self, s, noise_scale):
        with torch.no_grad():
            action = self.actor(torch.tensor(s).float().unsqueeze(0)).item()

            # Noise
            if noise_scale > 0.0:
                action += np.random.normal(0, noise_scale)
        
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

            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            target_a = (self.actor_target(s2) + noise).clamp(0,1)

            #Bellman
            target_q1 = self.critic_target1(s2, target_a)
            target_q2 = self.critic_target2(s2, target_a)
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