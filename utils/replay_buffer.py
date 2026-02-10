import random
import numpy as np

class ReplayBuffer:
    def __init__(self, size=1000000):
        self.buffer = []
        self.size = size

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, s2, d):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s2, d))


    def sample(self, batch):
        samples = random.sample(self.buffer, batch)
        return map(np.array, zip(*samples))