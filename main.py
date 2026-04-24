import numpy as np
from config import (SPOT, STRIKE, MATURITY, VOL, MU, DT, KAPPA, C, INIT_POSITION, R, TAU, GAMMA, LEARN_RATE, STATE_DIM, ACTION_DIM, HIDDEN_DIM, BATCH_SIZE,
                    ACTIONS_LIST, ACTION_DIMENSION, EPISODES, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD, PLOT, REPORT)

from env.hedging_env import HedgingEnv
from utils.bs import bs_delta, bs_price
from utils.compute_cost import compute_cost
from utils.generate_report import build_report
from utils.print import (print_hedge_table, plot_histogram, plot_learningcurve, plot_learningcurve_grid, 
                         plot_policy_heatmaps, plot_hedge_trajectory, plot_hybrid_decomposition,)
from utils.policy import (make_policy_BSM, make_policy_DDPG, make_policy_TD3, make_policy_DQN, make_policy_Hybrid,)

from train.train_DDPG_TD3 import train_DDPG_TD3, train_DDPG_TD3_without_OU_noise
from train.train_DQN import train_DQN
from train.train_hybrid import train_hybrid, train_hybrid_sequential

from models.dqn_agent import DQNAgent
from models.td3_agent import TD3Agent
from models.ddpg_agent import DDPGAgent
from models.hybrid_agent import HybridAgent

np.random.seed(0)
#torch.manual_seed(0)

# Environment
env = HedgingEnv(SPOT, STRIKE, MATURITY, VOL, MU, DT, KAPPA, C, INIT_POSITION, R)

# Single agents
dqn_agent = DQNAgent(STATE_DIM, ACTION_DIMENSION, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
ddpg_agent = DDPGAgent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
td3_agent = TD3Agent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)

# ddpg_agent_no_ou = DDPGAgent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
# td3_agent_no_ou = TD3Agent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)

# Hybrid agent sequential
# hybrid_dqn_sequential = DQNAgent(STATE_DIM, ACTION_DIMENSION, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
# hybrid_td3_sequential = TD3Agent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
# hybrid_agent_sequential = HybridAgent(hybrid_dqn_sequential, hybrid_td3_sequential, ACTIONS_LIST)

# Hybrid agent 
hybrid_dqn = DQNAgent(STATE_DIM, ACTION_DIMENSION, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
hybrid_td3 = TD3Agent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
hybrid_agent = HybridAgent(hybrid_dqn, hybrid_td3, ACTIONS_LIST)


# # Train without ou noise
# episode_rewards_DDPG_no_ou = train_DDPG_TD3_without_OU_noise(EPISODES, env, ddpg_agent_no_ou, BATCH_SIZE, min_noise = 0.01, noise_scale = 0.02, noise_decay = 0.9995, score_window_length=SCORE_WINDOW_LENGTH, stop_avg_reward=STOP_AVG_REWARD)
# episode_rewards_TD3_no_ou = train_DDPG_TD3_without_OU_noise(EPISODES, env, td3_agent_no_ou, BATCH_SIZE, min_noise = 0.01, noise_scale = 0.02, noise_decay = 0.9995, score_window_length=SCORE_WINDOW_LENGTH, stop_avg_reward=STOP_AVG_REWARD)

# Train Agents
episode_rewards_HYBRID = train_hybrid(EPISODES, env, hybrid_agent, BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)
# episode_rewards_HYBRID_sequential = train_hybrid_sequential(EPISODES, EPISODES, env, hybrid_agent_sequential, BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)
episode_rewards_DQN = train_DQN(EPISODES, env, dqn_agent, BATCH_SIZE, ACTIONS_LIST, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)
episode_rewards_DDPG = train_DDPG_TD3(EPISODES, env, ddpg_agent, BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)
episode_rewards_TD3 = train_DDPG_TD3(EPISODES, env, td3_agent, BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)


# Create Policies
policy_BSM = make_policy_BSM(STRIKE, R, VOL)
policy_DQN = make_policy_DQN(dqn_agent, ACTIONS_LIST)
#policy_DDPG_no_ou = make_policy_DDPG(ddpg_agent_no_ou)
#policy_TD3_no_ou  = make_policy_TD3(td3_agent_no_ou)
policy_DDPG = make_policy_DDPG(ddpg_agent)
policy_TD3  = make_policy_TD3(td3_agent)
# policy_Hybrid_sequential = make_policy_Hybrid(hybrid_agent_sequential, ACTIONS_LIST)
policy_Hybrid = make_policy_Hybrid(hybrid_agent, ACTIONS_LIST)

#### Compute costs ####
n_trails = 1000
n_steps = int(MATURITY / DT)

# Black- Scholes as Benchmark
Cost_BSM = compute_cost(policy_BSM, n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)

#Deep reinforcement Agents
Cost_hybrid = compute_cost(policy_Hybrid, n_trails, n_steps,SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)
# Cost_hybrid_sequential = compute_cost(policy_Hybrid_sequential, n_trails, n_steps,SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)

Cost_DQN = compute_cost(policy_DQN,  n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)
Cost_DDPG = compute_cost(policy_DDPG, n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)
Cost_TD3 = compute_cost(policy_TD3,  n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)
# Cost_DDPG_no_ou = compute_cost(policy_DDPG_no_ou, n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)
# Cost_TD3_no_ou = compute_cost(policy_TD3_no_ou,  n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)

52
#### Report and Plots ###
OptionPrice = bs_price(SPOT, STRIKE, R, MATURITY, VOL)

#print_hedge_table(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice)




if REPORT:
    build_report(
        Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice,
        episode_rewards_DDPG, episode_rewards_DQN,
        episode_rewards_TD3, episode_rewards_HYBRID,
        output_path="hedging_report.html"
    )

if PLOT:
    plot_histogram(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid)
    plot_learningcurve(episode_rewards_DDPG, episode_rewards_DQN,
                       episode_rewards_TD3, episode_rewards_HYBRID)
    plot_learningcurve_grid(episode_rewards_DDPG, episode_rewards_DQN,
                            episode_rewards_TD3, episode_rewards_HYBRID)
    plot_policy_heatmaps(ddpg_agent, dqn_agent, td3_agent, hybrid_agent, ACTIONS_LIST, MATURITY, VOL)
    plot_hedge_trajectory(env, ddpg_agent, dqn_agent, td3_agent, hybrid_agent, ACTIONS_LIST, VOL)
    plot_hybrid_decomposition(hybrid_agent, ACTIONS_LIST, MATURITY)

print("Done!")





























# #####
# import pandas as pd
# import matplotlib.pyplot as plt

# # COLORS = {
# #     "BSM":    "#2196F3",
# #     "DDPG":   "#F44336",
# #     "DQN":    "#4CAF50",
# #     "TD3":    "#FF9800",
# #     "Hybrid": "#9C27B0",
# # }

# COLORS = {
#     "BSM":    "#2196F3",
#     "DDPG":   "#F44336",
#     "DDPG (No OU)":    "#4CAF50",
#     "TD3":    "#FF9800",
#     "TD3 (No OU)": "#9C27B0",
# }

# def print_hedge_table_hybrid(Cost_BSM, Cost_DDPG, Cost_DDPG_no_ou, Cost_TD3, Cost_TD3_no_ou, OptionPrice):
#     HedgeComp = pd.DataFrame(
#         {
#             "BSM": 100 * np.array([
#                 -np.mean(Cost_BSM),
#                 np.std(Cost_BSM)
#             ]) / OptionPrice,

#             "DDPG": 100 * np.array([
#                 -np.mean(Cost_DDPG),
#                 np.std(Cost_DDPG)
#             ]) / OptionPrice,

#             "DDPG (No OU)": 100 * np.array([
#                 -np.mean(Cost_DDPG_no_ou),
#                 np.std(Cost_DDPG_no_ou)
#             ]) / OptionPrice,

#             "TD3": 100 * np.array([
#                 -np.mean(Cost_TD3),
#                 np.std(Cost_TD3)
#             ]) / OptionPrice,

#             "TD3 (No OU)": 100 * np.array([
#                 -np.mean(Cost_TD3_no_ou),
#                 np.std(Cost_TD3_no_ou)
#             ]) / OptionPrice,

#         },
#         index=[
#             "Average Hedge Cost (% of Option Price)",
#             "STD Hedge Cost (% of Option Price)"
#         ]
#     )

#     print(HedgeComp)

# def plot_histogram_hybrid(Cost_BSM, Cost_DDPG, Cost_DDPG_no_ou, Cost_TD3, Cost_TD3_no_ou):
#     fig, ax = plt.subplots(figsize=(11, 5))

#     costs = {
#         "BSM": Cost_BSM,
#         "DDPG": Cost_DDPG,
#         "DDPG (No OU)": Cost_DDPG_no_ou,
#         "TD3": Cost_TD3,
#         "TD3 (No OU)": Cost_TD3_no_ou,
#     }

#     all_vals = np.concatenate([-c * 100 for c in costs.values()])
#     bins = np.linspace(
#         np.percentile(all_vals, 1),
#         np.percentile(all_vals, 99),
#         40
#     )

#     for name, cost in costs.items():
#         ax.hist(-cost * 100, bins=bins,
#                 alpha=0.45, color=COLORS[name], label=name, edgecolor="none")
#         ax.axvline(-np.mean(cost) * 100,
#                 color=COLORS[name], linewidth=1.8, linestyle="--")

#     ax.set_xlabel("Hedging cost", fontsize=12)
#     ax.set_ylabel("Number of trials", fontsize=12)
#     ax.set_title("Distribution of hedging costs across 1,000 simulated paths", fontsize=13)
#     ax.legend(fontsize=10)
#     ax.grid(True, alpha=0.25, axis="y")
#     plt.tight_layout()
#     plt.savefig("plot_histogram.png", dpi=150, bbox_inches="tight")
#     plt.show()


# def plot_learningcurve(rewards_DDPG_no_ou, rewards_TD3_no_ou, rewards_DDPG, rewards_TD3,
#                     window=100):
#     """
#     Rolling mean learning curves for all four agents on one plot.
#     """
#     fig, ax = plt.subplots(figsize=(11, 5))

#     all_rewards = {
#         "DDPG (No OU)": rewards_DDPG_no_ou,
#         "TD3 (No OU)": rewards_TD3_no_ou,
#         "DDPG": rewards_DDPG,
#         "TD3": rewards_TD3,
#     }

#     for name, rewards in all_rewards.items():
#         s = pd.Series(rewards)
#         ax.plot(s.values, alpha=0.12, color=COLORS[name], linewidth=0.8)
#         ax.plot(s.rolling(window=window, min_periods=1).mean(),
#                 label=f"{name} ({window}-ep avg)",
#                 color=COLORS[name], linewidth=2)

#     ax.set_xlabel("Episode", fontsize=12)
#     ax.set_ylabel("Total episode reward", fontsize=12)
#     ax.set_title("Learning curves - all agents", fontsize=13)
#     ax.legend(fontsize=10)
#     ax.grid(True, alpha=0.25)
#     plt.tight_layout()
#     plt.savefig("plot_learningcurve_all.png", dpi=150, bbox_inches="tight")
#     plt.show()

    
# print_hedge_table_hybrid(Cost_BSM, Cost_DDPG, Cost_DDPG_no_ou, Cost_TD3, Cost_TD3_no_ou, OptionPrice)
# plot_histogram_hybrid(Cost_BSM, Cost_DDPG, Cost_DDPG_no_ou, Cost_TD3, Cost_TD3_no_ou)
# plot_learningcurve(episode_rewards_DDPG_no_ou, episode_rewards_TD3_no_ou, episode_rewards_DDPG, episode_rewards_TD3)
