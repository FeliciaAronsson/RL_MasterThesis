from env.hedging_env import HedgingEnv
import numpy as np
from utils.bs import bs_delta
from utils.replay_buffer import ReplayBuffer

variables = {
    "SpotPrice": 100,
    "Strike": 100,
    "Maturity": 21*3/250,
    "vol": 0.2,
    "mu": 0.05,
    "dT": 1/250,
    "kappa": 0.0 ,
    "c": 0,
    "InitPosition": 0,
    "r": 0.0
}

np.random.seed(0)

env = HedgingEnv(100, 100, 21*3/250, 0.2, 0.05, 1/250, 0.0, 0, 0, 0)

#initial_state = env.reset()
#print(initial_state)

testStep = env.step(1)
print(testStep)

print(bs_delta(50, 50, 0.1, 0.25, 0.3))

import torch
import numpy as np
from models.ddpg_agent import DDPGAgent


# 1. Inställningar (Hyperparametrar)
state_dim = 3
action_dim = 1

agent = DDPGAgent(state_dim, action_dim, 64, 0.1,0.001,0.0001)
episodes = 100
batch_size = 64
max_steps = int(21*30/250)

# 2. Träningsloopen
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # Agenten väljer en handling (t.ex. hur mycket delta-hedge)
        action = agent.select(state)
        
        # Miljön reagerar
        reward, next_state, done = env.step(action)
        
        # Spara erfarenheten i minnet (Replay Buffer)
        agent.buffer.add(state, action, reward, next_state, done)
        

        agent.train(batch_size)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    print(f"Episode: {episode}, Reward: {episode_reward:.2f}")


