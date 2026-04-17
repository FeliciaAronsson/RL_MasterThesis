import torch
import numpy as np


class HybridAgent:
    def __init__(self, dqn_agent, td3_agent, actions_list):
        self.dqn = dqn_agent
        self.td3 = td3_agent
        self.actions_list = actions_list 

    def select(self, state):
        """
        Select a hedge ratio using the hierarchical DQN + TD3 strategy.

        Step 1: DQN selects a bin index, defining a coarse interval
                [lower_bound, upper_bound] in the action space.
        Step 2: TD3 outputs a continuous value in [0, 1], which is rescaled
                to lie within the DQN-selected interval.
        Step 3: The final action is clipped to [0, 1] as a safety measure.

        Returns all three values so that the training loop can store the
        correct action in each sub-agent's replay buffer:
            - action:   the final hedge ratio sent to the environment
            - bin_idx:  what DQN chose (stored in DQN's buffer)
            - raw_td3:  what TD3 chose (stored in TD3's buffer)

        :param state: Current environment state as a numpy array.
        :return:      Tuple of (action, bin_idx, raw_td3).
        """
        # Step 1: DQN selects coarse bin
        bin_idx = self.dqn.select(state)
        lower_bound = self.actions_list[bin_idx]
        upper_bound = (
            self.actions_list[bin_idx + 1]
            if bin_idx + 1 < len(self.actions_list)
            else 1.0
        )

        # Step 2: TD3 selects fine-grained value within the bin
        raw_td3 = self.td3.select(state)

        # Step 3: Rescale TD3 output to [lower_bound, upper_bound]
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


    # def select(self, state):
    #     """Används aldrig, men tanken är att den ska kombinera DQN:s diskreta val med TD3:s finjustering."""
    #     # 1. Få grov-intervallet från DQN
    #     # Vi antar att dqn.select returnerar indexet för handlingen
    #     dqn_action_idx = self.dqn.select(state)
    #     base_hedge = self.discrete_actions[dqn_action_idx]

    #     # 2. Skapa ett utökat tillstånd för TD3
    #     # Vi lägger till DQN:s val som en "hint" i inputen
    #     #augmented_state = np.append(state, base_hedge)
    #     state[2] = base_hedge
    #     raw_delta = self.td3.select_ou(state)
        
    #     # 3. Få finjustering från TD3
    #     # TD3 bör tränas att ge ett litet delta, t.ex. mellan -0.05 och 0.05
    #      # 4. TRANSFORMATION (Viktigt!):
    #     # Vi centrerar 0.5 till att bli 0. 
    #     # Ett värde på 0.6 blir +0.02, ett värde på 0.4 blir -0.02.
    #     fine_tune = (raw_delta - 0.5) * 0.2  # Ger max +/- 0.1 i justering


    #     # 4. Kombinera och klicka (clip) mellan 0 och 1
    #     final_action = np.clip(base_hedge + fine_tune, 0.0, 1.0)
        
    #     return final_action
    


    #def train(self, batch_size):
    #    # Träna båda agenterna parallellt från deras respektive erfarenheter
    #    self.dqn.train(batch_size)
    #    self.td3.train(batch_size)