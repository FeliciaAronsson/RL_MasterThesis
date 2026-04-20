import numpy as np


class HybridAgent:
    def __init__(self, dqn_agent, td3_agent, actions_list):
        self.dqn = dqn_agent
        self.td3 = td3_agent
        self.actions_list = actions_list 

    def select(self, state):
        # DQN selects coarse bin
        bin_idx = self.dqn.select(state)
        lower_bound = self.actions_list[bin_idx]
        upper_bound = (
            self.actions_list[bin_idx + 1]
            if bin_idx + 1 < len(self.actions_list)
            else 1.0
        )

        # TD3 selects fine-grained value within the bin
        raw_td3 = self.td3.select(state)

        # Rescale TD3 output to [lower_bound, upper_bound]
        action = lower_bound + raw_td3 * (upper_bound - lower_bound)
        action = float(np.clip(action, 0.0, 1.0))

        return action, bin_idx, raw_td3

    def train(self, batch_size):
        self.dqn.train(batch_size)
        self.td3.train(batch_size)

    def reset_noise(self):
        self.td3.reset_noise()

    def __str__(self):
            return "Hybrid (DQN + TD3)"

