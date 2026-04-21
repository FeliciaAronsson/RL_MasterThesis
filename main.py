import numpy as np
from config import (SPOT, STRIKE, MATURITY, VOL, MU, DT, KAPPA, C, INIT_POSITION, R, TAU, GAMMA, LEARN_RATE, STATE_DIM, ACTION_DIM, HIDDEN_DIM, BATCH_SIZE,
                    ACTIONS_LIST, ACTION_DIMENSION, EPISODES, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD, PLOT, REPORT)

from env.hedging_env import HedgingEnv
from utils.bs import bs_delta, bs_price
from utils.compute_cost import compute_cost
from utils.generate_report import build_report
from utils.print import (print_hedge_table, plot_histogram, plot_cost_bars, plot_learningcurve, plot_learningcurve_grid, 
                         plot_policy_heatmaps, plot_hedge_trajectory, plot_hybrid_decomposition,)
from utils.policy import (make_policy_BSM, make_policy_DDPG, make_policy_TD3, make_policy_DQN, make_policy_Hybrid)

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

# Hybrid agent 
hybrid_dqn = DQNAgent(STATE_DIM, ACTION_DIMENSION, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
hybrid_td3 = TD3Agent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
hybrid_agent = HybridAgent(hybrid_dqn, hybrid_td3, ACTIONS_LIST)

# # Train without ou noise
#episode_rewards_DDPG = train_DDPG_TD3_without_OU_noise(EPISODES, env, ddpg_agent, BATCH_SIZE, min_noise = 0.01, noise_scale = 0.02, noise_decay = 0.9995, score_window_length=SCORE_WINDOW_LENGTH, stop_avg_reward=STOP_AVG_REWARD)
#episode_rewards_TD3 = train_DDPG_TD3_without_OU_noise(EPISODES, env, td3_agent, BATCH_SIZE, min_noise = 0.01, noise_scale = 0.02, noise_decay = 0.9995, score_window_length=SCORE_WINDOW_LENGTH, stop_avg_reward=STOP_AVG_REWARD)

# Train Agents
episode_rewards_DQN = train_DQN(EPISODES, env, dqn_agent, BATCH_SIZE, ACTIONS_LIST, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)
episode_rewards_DDPG = train_DDPG_TD3(EPISODES, env, ddpg_agent, BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)
episode_rewards_TD3 = train_DDPG_TD3(EPISODES, env, td3_agent, BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)
episode_rewards_HYBRID = train_hybrid(EPISODES, env, hybrid_agent, BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)

# Create Policies
policy_BSM = make_policy_BSM(STRIKE, R, VOL)
policy_DQN = make_policy_DQN(dqn_agent, ACTIONS_LIST)
policy_DDPG = make_policy_DDPG(ddpg_agent)
policy_TD3  = make_policy_TD3(td3_agent)
policy_Hybrid = make_policy_Hybrid(hybrid_agent, ACTIONS_LIST)

#### Compute costs ####
n_trails = 1000
n_steps = int(MATURITY / DT)

# Black- Scholes as Benchmark
Cost_BSM = compute_cost(policy_BSM, n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)

#Deep reinforcement Agents
Cost_DQN = compute_cost(policy_DQN,  n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)
Cost_DDPG = compute_cost(policy_DDPG, n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)
Cost_TD3 = compute_cost(policy_TD3,  n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)
Cost_hybrid = compute_cost(policy_Hybrid, n_trails, n_steps,SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)

#### Report and Plots ###
OptionPrice = bs_price(SPOT, STRIKE, R, MATURITY, VOL)

print_hedge_table(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice)
"""
if REPORT:
    build_report(
        Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice,
        episode_rewards_DDPG, episode_rewards_DQN,
        episode_rewards_TD3, episode_rewards_HYBRID,
        output_path="hedging_report.html"
    )
"""
if PLOT:
    plot_histogram(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice)
    plot_cost_bars(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice)
    plot_learningcurve(episode_rewards_DDPG, episode_rewards_DQN,
                       episode_rewards_TD3, episode_rewards_HYBRID)
    plot_learningcurve_grid(episode_rewards_DDPG, episode_rewards_DQN,
                            episode_rewards_TD3, episode_rewards_HYBRID)
    plot_policy_heatmaps(ddpg_agent, dqn_agent, td3_agent, hybrid_agent, ACTIONS_LIST, MATURITY, VOL)
    plot_hedge_trajectory(env, ddpg_agent, dqn_agent, td3_agent, hybrid_agent, ACTIONS_LIST, VOL)
    plot_hybrid_decomposition(hybrid_agent, ACTIONS_LIST, MATURITY)

print("Done!")
