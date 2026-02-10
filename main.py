from env.hedging_env import HedgingEnv
import numpy as np

# DiscountFactor, gammma = 0.9995
# TargetSmoothFactor, tau

variables = {
    "SpotPrice": 100,
    "Strike": 100,
    "Maturity": 21*3/250,
    "vol": 0.2,
    "mu": 0.05,
    "dT": 1/250,
    "kappa": 0.01,
    "c": 1.5,
    "InitPosition": 0,
    "r": 0.0,
    "gamma": 0.9995,
    "tau": 5e-4
}

np.random.seed(0)
                #(spot, strike, maturity, vol, mu, dT, kappa, c, initPosition, r):
env = HedgingEnv(100, 100, 21*3/250, 0.2, 0.05, 1/250, 0.01, 1.5, 0, 0)

from collections import deque
from models.agent import DDPGAgent
from models.agent import soft_update
import torch
import torch.nn as nn
from utils.replay_buffer import ReplayBuffer
max_episodes = 50
max_steps = 10   # maturity/dT
gamma = 0.9995
tau = 5e-4
obs_dim = 3
act_dim = 1
hidden_dim = 64
batch_size = 256


score_window = deque(maxlen=200)
stop_avg_reward = -40

agent = DDPGAgent(obs_dim, act_dim, hidden_dim, tau, gamma)

for episode in range(max_episodes):

    state = env.reset()
    episode_reward = 0.0

    for step in range(max_steps):

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # --- ACT ---
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy()[0]

        action = float(action)  # ← THIS IS THE FIX
        action += np.random.normal(0, 0.1) #, size=act_dim)  # exploration noise
        reward, next_state, done = env.step(action)
        

        agent.buffer.add(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state

        # --- LEARN ---
        if len(agent.buffer) > batch_size:
            batch = ReplayBuffer.sample(batch_size)

            s, a, r, s_next, d = batch

            # ---- Critic update ----
            with torch.no_grad():
                a_next = agent.target_actor(s_next)
                q_target = r + gamma * (1 - d) * agent.target_critic(s_next, a_next)

            q_current = agent.critic(s, a)
            critic_loss = nn.MSELoss()(q_current, q_target)

            agent.critic_opt.zero_grad()
            critic_loss.backward()
            agent.critic_opt.step()

            # ---- Actor update ----
            actor_loss = -agent.critic(s, agent.actor(s)).mean()

            agent.actor_opt.zero_grad()
            actor_loss.backward()
            agent.actor_opt.step()

            # ---- Target update ----
            soft_update(agent.target_actor, agent.actor, tau)
            soft_update(agent.target_critic, agent.critic, tau)

        if done:
            break

    # ---- Logging & stopping ----
    score_window.append(episode_reward)
    avg_reward = np.mean(score_window)

    print(f"Episode {episode}, Reward {episode_reward:.2f}, Avg {avg_reward:.2f}")

    if avg_reward > stop_avg_reward and len(score_window) == score_window.maxlen:
        print("Stopping: Average reward threshold reached")
        break



