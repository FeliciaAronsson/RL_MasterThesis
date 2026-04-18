import numpy as np
import torch 

from env.hedging_env import HedgingEnv

#from utils.policy import Policy
from utils.bs import bs_delta, bs_price
from utils.compute_cost import compute_cost
#from utils.print import plot_learningcurve, plot_histogram, print_hedge_table, plot_learningcurve_DDPG, plot_learningcurve_DQN, plot_learningcurve_TD3, plot_learningcurve_hybrid, plot_policy_heatmap, plot_hedge_trajectory, plot_hybrid_decomposition
from utils.generate_report import build_report
from utils.generate_report import print_hedge_table

from train.train_DDPG_TD3 import train_DDPG_TD3
from train.train_DQN import train_DQN
from train.train_hybrid import train_hybrid, train_hybrid_sequential

from models.dqn_agent import DQNAgent
from models.td3_agent import TD3Agent
from models.ddpg_agent import DDPGAgent
from models.hybrid_agent import HybridAgent

np.random.seed(0)

# Settings
spot = 100
strike = 100
maturity = 21*3/250
vol = 0.2
mu = 0.05
dT = 1/250
kappa = 0.01
c = 1.5
init_position = 0
r = 0

# Hyperparameters
tau = 5e-4
gamma = 0.9995
learnRate = 1e-4

# Neural Network settings
state_dim = 3
action_dim = 1
hidden_dim = 64
batch_size = 64

# To handle discrete actions for DQN 
actions_list = np.linspace(0, 1, 11)
#action_bins = [[actions_list[i], actions_list[i+1]] for i in range(len(actions_list)-1)]
action_dimension = len(actions_list)

# Define enviroment and agent
env = HedgingEnv(spot, strike, maturity, vol, mu, dT, kappa, c, init_position, r)

# Hybrid Agent
hybrid_dqn = DQNAgent(state_dim, action_dimension, hidden_dim, tau, gamma, learnRate)
hybrid_td3 = TD3Agent(state_dim, action_dim, hidden_dim, tau, gamma, learnRate)
hybrid_agent = HybridAgent(hybrid_dqn, hybrid_td3, actions_list)

# Agents
ddpg_agent = DDPGAgent(state_dim, action_dim, hidden_dim, tau, gamma, learnRate)
dqn_agent = DQNAgent(state_dim, action_dimension, hidden_dim, tau, gamma, learnRate)
td3_agent = TD3Agent(state_dim, action_dim, hidden_dim, tau, gamma, learnRate)

# Stopping criterion
score_window_lenght = 200
stop_avg_reward = 0
episodes = 500

# Variables to add noise (increase exploration)
noise_scale = 0.2 
noise_decay =  0.9995
min_noise = 0.01

# Train without ou noise
#episode_rewards_DDPG = train_RL(episodes, env, ddpg_agent, batch_size, min_noise, noise_scale, noise_decay, score_window, stop_avg_reward)
#episode_rewards_TD3 = train_RL(episodes, env, td3_agent, batch_size, min_noise, noise_scale, noise_decay, score_window, stop_avg_reward)

# Train with ou noise
episode_rewards_TD3 = train_DDPG_TD3(episodes, env, td3_agent, batch_size, score_window_lenght, stop_avg_reward)
episode_rewards_DDPG = train_DDPG_TD3(episodes, env, ddpg_agent, batch_size, score_window_lenght, stop_avg_reward)
episode_rewards_HYBRID = train_hybrid(episodes, env, hybrid_agent, batch_size, score_window_lenght, stop_avg_reward)

episode_rewards_DQN = train_DQN(episodes, env, dqn_agent, batch_size, actions_list, score_window_lenght, stop_avg_reward)


# Cost function
n_trails = 1000
n_steps = int(maturity / dT)

def policy_BSM(mR, TTM, Pos):
    """
    Docstring for policy_BSM
    
    :param mR: Description
    :param TTM: Description
    :param Pos: Description
    """
    S = mR * strike
    return bs_delta(S, strike, r, TTM, vol)


def policy_DDPG(mR, TTM, Pos):
    """
    Docstring for policy_RL
    
    :param mR: Description
    :param TTM: Description
    :param Pos: Description
    """
    
    state = np.stack([mR, TTM, Pos], axis=1)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        action = ddpg_agent.actor(state_tensor).cpu().numpy()

    return action.squeeze()

def policy_TD3(mR, TTM, Pos):
    """
    Docstring for policy_RL
    
    :param mR: Description
    :param TTM: Description
    :param Pos: Description
    """
    
    state = np.stack([mR, TTM, Pos], axis=1)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        action = td3_agent.actor(state_tensor).cpu().numpy()

    return action.squeeze()

def policy_DQN(mR, TTM, Pos):
    state = np.stack([mR, TTM, Pos], axis=1)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():                        # Här är policyn argmax som ända skillnaden
        action_index = dqn_agent.qnet(state_tensor).argmax(dim=1).cpu().numpy()

    return actions_list[action_index]

def policy_Hybrid(mR, TTM, Pos):
    state = np.stack([mR, TTM, Pos], axis=1)
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        bin_idx = hybrid_dqn.qnet(state_tensor).argmax(dim=1).cpu().numpy()
        raw_td3 = hybrid_td3.actor(state_tensor).cpu().numpy().squeeze()
    lower = actions_list[bin_idx]
    upper = np.where(bin_idx + 1 < len(actions_list),
                     actions_list[np.minimum(bin_idx + 1, len(actions_list)-1)],
                     1.0)
    return np.clip(lower + raw_td3 * (upper - lower), 0.0, 1.0)
# Calculate the cost

# Black- Scholes as Benchmark
Cost_BSM = compute_cost(policy_BSM, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)

#Deep reinforcement Agents
Cost_DQN = compute_cost(policy_DQN,  n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
Cost_DDPG = compute_cost(policy_DDPG, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
Cost_TD3 = compute_cost(policy_TD3,  n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
Cost_hybrid = compute_cost(policy_Hybrid, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)

OptionPrice = bs_price(spot, strike, r, maturity, vol)

build_report(
    Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice,
    episode_rewards_DDPG, episode_rewards_DQN,
    episode_rewards_TD3, episode_rewards_HYBRID,
    output_path="hedging_report.html"
)

print_hedge_table(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice)

print("Done!")
