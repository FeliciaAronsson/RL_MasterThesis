import torch
import numpy as np


class HybridAgent:
    def __init__(self, dqn_agent, td3_agent, discrete_actions):
        self.dqn = dqn_agent
        self.td3 = td3_agent
        # Lista på diskreta värden, t.ex. [0.0, 0.1, ..., 1.0]
        self.discrete_actions = discrete_actions 


    def select(self, state):
        """Används aldrig, men tanken är att den ska kombinera DQN:s diskreta val med TD3:s finjustering."""
        # 1. Få grov-intervallet från DQN
        # Vi antar att dqn.select returnerar indexet för handlingen
        dqn_action_idx = self.dqn.select(state)
        base_hedge = self.discrete_actions[dqn_action_idx]

        # 2. Skapa ett utökat tillstånd för TD3
        # Vi lägger till DQN:s val som en "hint" i inputen
        #augmented_state = np.append(state, base_hedge)
        state[2] = base_hedge
        raw_delta = self.td3.select_ou(state)
        
        # 3. Få finjustering från TD3
        # TD3 bör tränas att ge ett litet delta, t.ex. mellan -0.05 och 0.05
         # 4. TRANSFORMATION (Viktigt!):
        # Vi centrerar 0.5 till att bli 0. 
        # Ett värde på 0.6 blir +0.02, ett värde på 0.4 blir -0.02.
        fine_tune = (raw_delta - 0.5) * 0.2  # Ger max +/- 0.1 i justering


        # 4. Kombinera och klicka (clip) mellan 0 och 1
        final_action = np.clip(base_hedge + fine_tune, 0.0, 1.0)
        
        return final_action
    


    #def train(self, batch_size):
    #    # Träna båda agenterna parallellt från deras respektive erfarenheter
    #    self.dqn.train(batch_size)
    #    self.td3.train(batch_size)