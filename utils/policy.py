

from utils.bs import bs_delta
import torch
import numpy as np


def policy_BSM(mR, TTM, Pos, strike, r, vol):
    """
    Policy for Black Scholes hedging
    
    :param mR: Spot/Strike
    :param TTM: Time to maturity
    :param Pos: Position 
    :param strike: Strike Price
    :param r: Risk free rate
    :param vol: Expected volatility
    """
     
    S = mR * strike
    return bs_delta(S, strike, r, TTM, vol)


def policy_RL(spot, strike, TTM, Pos, agent): 
    """
    Policy for reinforcement learning hedging
    
    :param mR: Description
    :param TTM: Time to maturity
    :param Pos: Position
    :param agent: Agent
    """
    mR = spot/strike
    
    state = np.stack([mR, TTM, Pos], axis=1)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        action = agent.actor(state_tensor).cpu().numpy()

    return action.squeeze()