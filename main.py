from env.hedging_env import HedgingEnv
import numpy as np
from utils.bs import bs_delta
import numpy as np
from models.ddpg_agent import DDPGAgent
from collections import deque
from utils.compute_cost import compute_cost
from policy.policy import policy_BSM
np.random.seed(0)

# Settings
spot = int(100)
strike = 100
maturity = 21*3/250
vol = 0.2
mu = 0.05
dT = 1/250
kappa = 0
c = 0
init_position = 0
rf_rate = 0

n_step = maturity/dT

state_dim = 3
action_dim = 1
hidden_dim = 64
tau = 0.1
gamma = 0.001
learnRate = 0.0001

episodes = 100
batch_size = 64
max_steps = int(21*30/250)

env = HedgingEnv(spot, strike, maturity, vol, mu, dT, kappa, c, init_position, rf_rate)
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

    print(f"Episode {episode}, Reward {episode_reward:.2f}, Avg {avg_reward:.2f}")

    if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
        print("Stopping: Average reward threshold reached")
        break

    # Policy
    #policy_rl = lambda state: agent.select(state[0])
    policy_bsm = lambda state: bs_delta(state[0]*spot, strike, rf_rate, max(state[1],1e-8), np.sqrt(vol))

    #policy_bsm = policy_BSM(state[0]*spot, strike, rf_rate, max(state[1],1e-8), np.sqrt(vol))
    #print(type(policy_bsm))

    #policy_bsm = policy_BSM(
    #    state[0] * spot,
    #    max(state[1], 1e-8),   # ttm
    #    init_position,              # pos
    #    rf_rate,
    #    np.sqrt(vol)
    #    )

    #print(type(policy_bsm))
    # Validate agent 

    n_trial = 50 
    n_step = 10
    Costs_BSM = compute_cost(policy_bsm, n_trial, n_step, spot, strike, maturity, rf_rate, vol, init_position, dT, mu, kappa)
    #Costs_RL = compute_cost(policy_rl, n_trial, n_step, spot, strike, maturity, rf_rate, vol, init_position, dT, mu, kappa)