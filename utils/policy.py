import numpy as np
import torch
from utils.bs import bs_delta


def make_policy_BSM(strike, r, vol):
    """
    Returns a BSM delta-hedging policy function.
    The returned function has the signature (mR, TTM, Pos) required by compute_cost.
    """
    def policy(mR, TTM, Pos):
        S = mR * strike
        return bs_delta(S, strike, r, TTM, vol)
    return policy


def make_policy_DDPG(agent):
    """
    Returns a DDPG policy function.
    The returned function has the signature (mR, TTM, Pos) required by compute_cost.
    """
    def policy(mR, TTM, Pos):
        state = np.stack([mR, TTM, Pos], axis=1)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy()
        return action.squeeze()
    return policy


def make_policy_TD3(agent):
    """
    Returns a TD3 policy function.
    The returned function has the signature (mR, TTM, Pos) required by compute_cost.
    """
    def policy(mR, TTM, Pos):
        state = np.stack([mR, TTM, Pos], axis=1)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy()
        return action.squeeze()
    return policy


def make_policy_DQN(agent, actions_list):
    """
    Returns a DQN policy function.
    The returned function has the signature (mR, TTM, Pos) required by compute_cost.
    """
    def policy(mR, TTM, Pos):
        state = np.stack([mR, TTM, Pos], axis=1)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_index = agent.qnet(state_tensor).argmax(dim=1).cpu().numpy()
        return actions_list[action_index]
    return policy


def make_policy_Hybrid(hybrid_agent, actions_list):
    """
    Returns a Hybrid (DQN + TD3) policy function.
    The returned function has the signature (mR, TTM, Pos) required by compute_cost.
    """
    def policy(mR, TTM, Pos):
        state = np.stack([mR, TTM, Pos], axis=1)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            bin_idx = hybrid_agent.dqn.qnet(state_tensor).argmax(dim=1).cpu().numpy()
            raw_td3 = hybrid_agent.td3.actor(state_tensor).cpu().numpy().squeeze()
        lower = actions_list[bin_idx]
        upper = np.where(
            bin_idx + 1 < len(actions_list),
            actions_list[np.minimum(bin_idx + 1, len(actions_list) - 1)],
            1.0
        )
        return np.clip(lower + raw_td3 * (upper - lower), 0.0, 1.0)
    return policy