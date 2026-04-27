import random
import numpy as np

class ExperienceBuffer:
    """
    The ExperienceBuffer class implements a replay buffer for storing and sampling transitions in reinforcement learning.
    It allows the agent to store past experiences (state, action, reward, next_state, done) and sample random batches of these experiences for training the neural networks.

    Collecive experience
    """
    def __init__(self, size=1_000_000):
        self.buffer = []
        self.size = size

    def add(self, state, action_agent1, action_agent2, reward, next_state, done):
        """
        Add the transition to the agents memory bank (the Replay Buffer)
        
        :param state: Current state
        :param action_agent1: Action (0,1)
        :param action_agent2: Action (0,1)
        :param reward: Reward
        :param next_state: Next state
        :param done: Done if time to maturity (TTM) is reached
        """
    
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append((state, action_agent1, action_agent2, reward, next_state, done))


    # def sample(self, batch):
    #     """
    #     Train the agents network (Actor-Critic) on a random barch
    #     """
    #     samples = random.sample(self.experience_buffer, batch)
    #     state, action_agent1, action_agent2, reward, next_state, done =  map(np.array, zip(*samples))
        
    #     return state, action_agent1, action_agent2, reward, next_state, done

    def sample_for_dqn(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        # Packa upp, men ge DQN bara action_agent1 (indexet)
        s, a1, a2, r, s2, d = zip(*samples)
        return map(np.array, [s, a1, r, s2, d])

    def sample_for_td3(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        # Packa upp, men ge TD3 bara action_agent2 (kontinuerliga värdet)
        s, a1, a2, r, s2, d = zip(*samples)
        return map(np.array, [s, a2, r, s2, d])
    
    def __len__(self):
        return len(self.buffer)