
import numpy as np
from collections import deque
import torch 


from env.hedging_env import HedgingEnv
from models.ddpg_agent import DDPGAgent
from models.td3_agent import TD3Agent
from utils.bs import bs_delta, bs_price
from utils.compute_cost import compute_cost
#from utils.policy import policy_BSM, policy_RL
from train.train import train_RL
from utils.print import plot_learningcurve, plot_histogram, print_hedge_table

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

# Define enviroment and agent
env = HedgingEnv(spot, strike, maturity, vol, mu, dT, kappa, c, init_position, r)
#agent = DDPGAgent(state_dim, action_dim, hidden_dim, tau, gamma, learnRate)
agent = TD3Agent(state_dim, action_dim, hidden_dim, tau, gamma, learnRate)

# Stopping criterion
score_window = deque(maxlen=200)
stop_avg_reward = -40

# Variables to add noice (increase exploration)
noise_scale = 0.2
noise_decay = 0.9995
min_noise = 0.01

# Training
episodes = 1250
episode_rewards = train_RL(episodes, env, agent, batch_size, min_noise, noise_scale, noise_decay, score_window, stop_avg_reward)


# Cost function
n_trails = 1000
n_steps = int(maturity / dT)
mR = spot/strike
Pos = init_position

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
        action = agent.actor(state_tensor).cpu().numpy()

    return action.squeeze()


Costs_BSM = compute_cost(policy_BSM, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
Costs_RL = compute_cost(policy_RL, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
OptionPrice = bs_price(spot,strike,r,maturity,vol)


# Plot results
print_hedge_table(Costs_BSM, Costs_RL, OptionPrice)
plot_histogram(Costs_RL, Costs_BSM)
plot_learningcurve(episode_rewards)