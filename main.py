
import numpy as np
from collections import deque
import torch 


from env.hedging_env import HedgingEnv

from utils.policy import Policy
from utils.bs import bs_delta, bs_price
from utils.compute_cost import compute_cost

#from utils.policy import policy_BSM, policy_RL
from train.train import train_RL
from train.train_DQN import train_DQN
from utils.print import plot_learningcurve, plot_histogram, print_hedge_table

# Agents 
from models.dqn_agent import DQNAgent
from models.td3_agent import TD3Agent
from models.ddpg_agent import DDPGAgent

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
action_dimension = len(actions_list)

# Define enviroment and agent
env = HedgingEnv(spot, strike, maturity, vol, mu, dT, kappa, c, init_position, r)

# td3_agent = 
ddpg_agent = DDPGAgent(state_dim, action_dim, hidden_dim, tau, gamma, learnRate)
dqn_agent = DQNAgent(state_dim, action_dimension, hidden_dim, tau, gamma, learnRate)
td3_agent = TD3Agent(state_dim, action_dim, hidden_dim, tau, gamma, learnRate)

# Stopping criterion
score_window = deque(maxlen=200)
stop_avg_reward = 0
episodes = 1000

# Variables to add noice (increase exploration)
noise_scale = 0.2
noise_decay =  0.9995
min_noise = 0.01


episode_rewards_DDPG = train_RL(episodes, env, ddpg_agent, batch_size, min_noise, noise_scale, noise_decay, score_window, stop_avg_reward)
episode_rewards_TD3 = train_RL(episodes, env, td3_agent, batch_size, min_noise, noise_scale, noise_decay, score_window, stop_avg_reward)
episode_rewards_DQN = train_DQN(episodes, env, dqn_agent, batch_size, actions_list, score_window, stop_avg_reward)

# Cost function
n_trails = 1000
n_steps = int(maturity / dT)

#policy_dqn = Policy(dqn_agent, strike, vol, r, actions_list)
#policy_rl = Policy(ddpg_agent, strike, vol, r, actions_list )

def policy_BSM(mR, TTM, Pos):
    """
    Docstring for policy_BSM
    
    :param mR: Description
    :param TTM: Description
    :param Pos: Description
    """
    S = mR * strike
    return bs_delta(S, strike, r, TTM, vol)


def policy_RL(mR, TTM, Pos):
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

def policy_DQN(mR, TTM, Pos):
    state = np.stack([mR, TTM, Pos], axis=1)
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():                        # Här är policyn argmax som ända skillnaden
        action_index = dqn_agent.qnet(state_tensor).argmax(dim=1).cpu().numpy()

    return actions_list[action_index]


Cost_BSM = compute_cost(policy_BSM, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
Cost_DDPG = compute_cost(policy_RL, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
Cost_DQN = compute_cost(policy_DQN,  n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
Cost_TD3 = compute_cost(policy_RL,  n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)

OptionPrice = bs_price(spot, strike, r, maturity, vol)


# Plot results
#print_hedge_table(Cost_BSM, Cost_RL, OptionPrice)
#plot_histogram(Cost_RL, Cost_BSM)
#plot_learningcurve(episode_rewards)

print_hedge_table(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, OptionPrice)
plot_histogram(Cost_DDPG, Cost_DQN, Cost_TD3, Cost_BSM)
plot_learningcurve(episode_rewards_DDPG, episode_rewards_DQN, episode_rewards_TD3)