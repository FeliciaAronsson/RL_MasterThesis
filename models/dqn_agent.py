 
import torch
import torch.optim as optim
import torch.nn as nn
from models.q_networks import QNetwork
from utils.replay_buffer import ReplayBuffer
import numpy as np

class DQNAgent:
    def __init__(self, obs_dim, act_dim, hidden_dim, tau, gamma, learnRate, epsilon_start=1.0, epsilon_decay = 0.995, epsilon_min = 0.01):
        self.tau = tau
        self.gamma = gamma
        self.act_dim = act_dim

        # Epsilon greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q = QNetwork(obs_dim, act_dim, hidden_dim)
        self.q_target = QNetwork(obs_dim, act_dim, hidden_dim)

        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr= learnRate)
        
        self.buffer = ReplayBuffer()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q.to(self.device)
        self.q_target.to(self.device)

   
    def select(self, state, eps):
        if np.random.rand() < eps: 
            return np.random.randint(0, self.act_dim-1)
        
        state = torch.tensor(state).float().unsqueeze(0)
        self.q.eval()
        
        with torch.no_grad():
            action_values = self.q(state)

        self.q.train()
        return np.argmax(action_values.cpu().data.numpy)



    def train(self, batch_size):

        if len(self.buffer) < batch_size:
            return
        

        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        state = torch.tensor(state).float()
        action = torch.tensor(action).long().unsqueeze(1) #idx long for gather
        reward = torch.tensor(reward).float().unsqueeze(1)
        next_state = torch.tensor(next_state).float()
        done = torch.tensor(done).float().unsqueeze(1)


        q_target_next = self.q_target(next_state).detach().max(1)[0].unsqueeze(1)
        q_targets = reward + (self.gamma * q_target_next * (1-done))

        q_expected = self.q(state).gather(1, action)
        
        loss = nn.MSELoss()(q_expected, q_targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        #soft update 
        for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

            self.epsilon_start = max(self.epsilon_min, self.epsilon_start * self.epsilon_decay)
    

       