

from utils.bs import bs_delta
import torch
import numpy as np

class Policy():

    def __init__(self, agent, strike, vol, rate, actions_list):

        self.strike = strike
        self.vol = vol
        self.rate = rate
        self.agent = agent
        self.actions_list = actions_list


    def policy_BSM(self, mR, TTM, Pos):
        """
        Policy for Black Scholes hedging
        
        :param mR: Spot/Strike
        :param TTM: Time to maturity
        :param Pos: Position 
        :param strike: Strike Price
        :param r: Risk free rate
        :param vol: Expected volatility
        """
        
        S = mR * self.strike
        return bs_delta(S, self.strike, self.rate, TTM, self.vol)


    def policy_RL(self, mR, TTM, Pos): 
        """
        Policy for reinforcement learning hedging
        
        :param mR: Description
        :param TTM: Time to maturity
        :param Pos: Position
        :param agent: Agent
        """
        
        state = np.stack([mR, TTM, Pos], axis=1)
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action = self.agent.actor(state_tensor).cpu().numpy()

        return action.squeeze()
    
    def policy_DQN(self, mR, TTM, Pos):
        
        state = np.stack([mR, TTM, Pos], axis=1)
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():                        # Här är policyn argmax som ända skillnaden
            action_index = self.agent.qnet(state_tensor).argmax(dim=1).cpu().numpy()

        return self.actions_list[action_index]
    
