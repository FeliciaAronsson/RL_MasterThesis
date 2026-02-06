from env.hedging_env import HedgingEnv
import numpy as np

# def test_single_episode(env, fixed_action=0.5):
#     state = env.reset()
#     done = False

#     print("Step | Moneyness | TTM | Position | Reward")
#     print("-" * 50)

#     step = 0
#     total_reward = 0.0

#     while not done:
#         state, reward, done = env.step(fixed_action)
#         m, ttm, pos = state
#         total_reward += reward

#         print(f"{step:>4} | {m:.3f}     | {ttm:.3f} | {pos:.2f}     | {reward:.4f}")
#         step += 1

#     print("-" * 50)
#     print(f"Total episode reward: {total_reward:.4f}")

cfg = {
    "SpotPrice": 100,
    "Strike": 100,
    "Maturity": 21*3/250,
    "vol": 0.2,
    "mu": 0.05,
    "dT": 1/250,
    "kappa": 0.01,
    "c": 1.5,
    "InitPosition": 0,
    "r": 0.0
}

np.random.seed(0)

env = HedgingEnv(100, 100, 21*3/250, 0.2, 0.05, 1/250, 0.01, 1.5, 0, 0)

initialState = env.reset()
print(initialState)

testStep = env.step()
print(testStep)
# test_single_episode(env, fixed_action=0.5)