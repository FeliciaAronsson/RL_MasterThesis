from env.hedging_env import HedgingEnv
import numpy as np
from utils.bs import bs_delta
from utils.bs import bs_price
import numpy as np
from models.ddpg_agent import DDPGAgent
from collections import deque
from utils.compute_cost import compute_cost_rl
from utils.compute_cost import compute_cost_bsm
import torch

np.random.seed(3)

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

state_dim = 3
action_dim = 1
hidden_dim = 64
tau = 5e-4
gamma = 0.9995
learnRate = 1e-4

episodes = 5000
batch_size = 64
max_steps = int(21*30/250)

env = HedgingEnv(spot, strike, maturity, vol, mu, dT, kappa, c, init_position, r)
agent = DDPGAgent(state_dim, action_dim, hidden_dim, tau, gamma, learnRate)

score_window = deque(maxlen=200)
stop_avg_reward = -40

# Training
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        action = agent.select(state)
        reward, next_state, done = env.step(action)
        agent.buffer.add(state, action, reward, next_state, done)
    
        agent.train(batch_size)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break

    # Logging & stopping 
    score_window.append(episode_reward)
    avg_reward = np.mean(score_window)

    #print(f"Episode {episode}, Reward {episode_reward:.2f}, Avg {avg_reward:.2f}")

    if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
        print("Stopping: Average reward threshold reached")
        break

    # Policy
    #policy_rl = lambda state: agent.select(state)
    #policy_bsm = lambda action: bs_delta(state[0]*spot, strike, r, max(state[1],1e-8), np.sqrt(vol))



def policy_BSM(S, K, r, T, sigma):
    return bs_delta(S, K, r, T, sigma)

# def policy_RL(state):
    
#     return agent.select(state)



def policy_RL(mR, TTM, Pos):
    """P
    mR   : shape (nTrials,)
    TTM  : shape (nTrials,)
    Pos  : shape (nTrials,)
    """

    # Build state matrix: shape (nTrials, 3)
    state = np.stack([mR, TTM, Pos], axis=1)

    # Convert to torch
    state_tensor = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        action = agent.actor(state_tensor).cpu().numpy()

    return action.squeeze()

n_trails = 1000
n_steps = int(maturity / dT)
Costs_BSM = compute_cost_bsm(policy_BSM, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)
Costs_RL = compute_cost_rl(policy_RL, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)


S_test = np.array([90, 100, 110])
T_test = np.array([0.5, 0.1, 0.0])

print(bs_price(S_test, 100, 0.01, T_test, 0.2))
print(bs_delta(S_test, 100, 0.01, T_test, 0.2))

import pandas as pd

OptionPrice = bs_price(spot,strike,r,maturity,vol)

HedgeComp = pd.DataFrame(
    {
        "BSM": 100 * np.array([
            -np.mean(Costs_BSM),
            np.std(Costs_BSM)
        ]) / OptionPrice,

        "RL": 100 * np.array([
            -np.mean(Costs_RL),
            np.std(Costs_RL)
        ]) / OptionPrice
    },
    index=[
        "Average Hedge Cost (% of Option Price)",
        "STD Hedge Cost (% of Option Price)"
    ]
)

print(HedgeComp)




import matplotlib.pyplot as plt
import numpy as np

num_bins = 10

plt.figure()

plt.hist(-Costs_RL, bins=num_bins, color='red', alpha=0.5, label='RL Hedge')
plt.hist(-Costs_BSM, bins=num_bins, color='blue', alpha=0.5, label='Theoretical BLS Delta')

plt.xlabel('Hedging Costs')
plt.ylabel('Number of Trials')
plt.title('RL Hedge Costs vs. BLS Hedge Costs')
plt.legend(loc='best')

plt.show()

