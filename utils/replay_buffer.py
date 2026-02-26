import random
import numpy as np

class ReplayBuffer:
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

        :param batch: Batch size
        """
        samples = random.sample(self.buffer, batch)
        return map(np.array, zip(*samples))