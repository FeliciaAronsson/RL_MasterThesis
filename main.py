from env.hedging_env import HedgingEnv
import numpy as np
from utils.bs import bs_delta
from utils.bs import bs_price
import numpy as np
from models.ddpg_agent import DDPGAgent
from collections import deque
from utils.compute_cost import compute_cost

np.random.seed(0)

# Settings
spot = 100
strike = 100
maturity = 21*3/250
vol = 0.2
mu = 0.05
dT = 1/250
kappa = 0
c = 0
init_position = 0
r = 0

state_dim = 3
action_dim = 1
hidden_dim = 64
tau = 0.1
gamma = 0.001
learnRate = 0.0001

episodes = 100
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

    print(f"Episode {episode}, Reward {episode_reward:.2f}, Avg {avg_reward:.2f}")

    if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
        print("Stopping: Average reward threshold reached")
        break

    # Policy
    #policy_rl = lambda state: agent.select(state)
    #policy_bsm = lambda action: bs_delta(state[0]*spot, strike, r, max(state[1],1e-8), np.sqrt(vol))



    def policy_BSM(S, K, r, T, sigma):
        return bs_delta(S, K, r, T, sigma)
    
    def policy_RL(state):
        return agent.select(state)

    n_trails = 50
    n_steps = int(maturity / dT)
    #cost_bsm = compute_cost(policy_BSM, n_trails, n_steps, spot, strike, maturity, r, vol, init_position, dT, mu, kappa)


    S_test = np.array([90, 100, 110])
    T_test = np.array([0.5, -1, 0.0])

    print(bs_price(S_test, 100, 0.01, T_test, 0.2))
    print(bs_delta(S_test, 100, 0.01, T_test, 0.2))

    #print(cost_bsm)