import random
import numpy as np

class ReplayBuffer:
    """
    The ReplayBuffer class implements a replay buffer for storing and sampling transitions in reinforcement learning.
    It allows the agent to store past experiences (state, action, reward, next_state, done) and sample random batches of these experiences for training the neural networks.
    """
    def __init__(self, size=1_000_000):
        self.buffer = []
        self.size = size

    def add(self, state, action, reward, next_state, done):
        """
        Add the transition to the agents memory bank (the Replay Buffer)
        
        :param state: Current state
        :param action: Action (0,1)
        :param reward: Reward
        :param next_state: Next state
        :param done: Done if time to maturity (TTM) is reached
        """
    
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))


    def sample(self, batch):
        """
        Train the agents network (Actor-Critic) on a random barch
        """
        samples = random.sample(self.buffer, batch)
        state, action, reward, next_state, done =  map(np.array, zip(*samples))
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)