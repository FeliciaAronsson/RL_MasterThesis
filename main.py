import numpy as np
import torch
import os as os
from config import (SPOT, STRIKE, MATURITY, VOL, MU, DT, KAPPA, C, INIT_POSITION, R, TAU, GAMMA, LEARN_RATE, STATE_DIM, ACTION_DIM, HIDDEN_DIM, BATCH_SIZE,
                    ACTIONS_LIST, ACTION_DIMENSION, EPISODES, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD, PLOT, REPORT)

from env.hedging_env import HedgingEnv
from utils.bs import bs_delta, bs_price
from utils.compute_cost import compute_cost
from utils.generate_report import build_report
from utils.print import (print_hedge_table, plot_histogram, plot_learningcurve, plot_learningcurve_grid, 
                         plot_policy_3d, plot_hedge_trajectory,)
from utils.policy import (make_policy_BSM, make_policy_DDPG, make_policy_TD3, make_policy_DQN, make_policy_Hybrid)

from train.train_DDPG_TD3 import train_DDPG_TD3, train_DDPG_TD3_without_OU_noise
from train.train_DQN import train_DQN
from train.train_hybrid import train_hybrid, train_hybrid_sequential

from models.dqn_agent import DQNAgent
from models.td3_agent import TD3Agent
from models.ddpg_agent import DDPGAgent
from models.hybrid_agent import HybridAgent

# Choice of which agents to run and evaluate.
RUN_CONFIG = {
    "DQN":    True,
    "DDPG":   True,
    "TD3":    True,
    "Hybrid": True,
    "Hybrid Sequential": False,
}

np.random.seed(0)
env = HedgingEnv(SPOT, STRIKE, MATURITY, VOL, MU, DT, KAPPA, C, INIT_POSITION, R)
n_trails = 1000 
n_steps = int(MATURITY / DT)

agents = {}
rewards = {}
agent_costs = {}
OptionPrice = bs_price(SPOT, STRIKE, R, MATURITY, VOL)

save_path = 'saved_models'
if not os.path.exists(save_path):
    os.makedirs(save_path)

### Training ###
if RUN_CONFIG["DQN"]:
    agents["DQN"] = DQNAgent(STATE_DIM, ACTION_DIMENSION, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
    rewards["DQN"] = train_DQN(EPISODES, env, agents["DQN"], BATCH_SIZE, ACTIONS_LIST, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)
    torch.save(agents["DQN"].qnet.state_dict(), "saved_models/dqn_qnet.pth")

if RUN_CONFIG["DDPG"]:
    agents["DDPG"] = DDPGAgent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
    rewards["DDPG"] = train_DDPG_TD3(EPISODES, env, agents["DDPG"], BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)

    ## Train DDPG without OU noise for comparison
    # rewards["DDPG"] = train_DDPG_TD3_without_OU_noise(EPISODES, env, agents["DDPG"], BATCH_SIZE, min_noise = 0.01, noise_scale = 0.02, noise_decay = 0.9995, score_window_length=SCORE_WINDOW_LENGTH, stop_avg_reward=STOP_AVG_REWARD)

    torch.save(agents["DDPG"].actor.state_dict(), "saved_models/ddpg_actor.pth")

if RUN_CONFIG["TD3"]:
    agents["TD3"] = TD3Agent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
    rewards["TD3"] = train_DDPG_TD3(EPISODES, env, agents["TD3"], BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)

    ## Train TD3 without OU noise for comparison
    #rewards["TD3"] = train_DDPG_TD3_without_OU_noise(EPISODES, env, agents["TD3"], BATCH_SIZE, min_noise = 0.01, noise_scale = 0.02, noise_decay = 0.9995, score_window_length=SCORE_WINDOW_LENGTH, stop_avg_reward=STOP_AVG_REWARD)

    torch.save(agents["TD3"].actor.state_dict(), "saved_models/td3_actor.pth")

if RUN_CONFIG["Hybrid"]:
    hybrid_dqn = DQNAgent(STATE_DIM, ACTION_DIMENSION, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
    hybrid_td3 = TD3Agent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
    agents["Hybrid"] = HybridAgent(hybrid_dqn, hybrid_td3, ACTIONS_LIST)
    rewards["Hybrid"] = train_hybrid(EPISODES, env, agents["Hybrid"], BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)

if RUN_CONFIG["Hybrid Sequential"]:
    hybrid_dqn_sequential = DQNAgent(STATE_DIM, ACTION_DIMENSION, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
    hybrid_td3_sequential = TD3Agent(STATE_DIM, ACTION_DIM, HIDDEN_DIM, TAU, GAMMA, LEARN_RATE)
    agents["Hybrid Sequential"] = HybridAgent(hybrid_dqn_sequential, hybrid_td3_sequential, ACTIONS_LIST)
    rewards["Hybrid Sequential"] = train_hybrid_sequential(EPISODES, EPISODES, env, agents["Hybrid Sequential"], BATCH_SIZE, SCORE_WINDOW_LENGTH, STOP_AVG_REWARD)

### Evaluation ###
policy_BSM = make_policy_BSM(STRIKE, R, VOL)
agent_costs["BSM"] = compute_cost(policy_BSM, n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)

for name, agent in agents.items():
    if name == "DQN":
        p = make_policy_DQN(agent, ACTIONS_LIST)
    elif name == "DDPG":
        p = make_policy_DDPG(agent)
    elif name == "TD3":
        p = make_policy_TD3(agent)
    elif name == "Hybrid":
        p = make_policy_Hybrid(agent, ACTIONS_LIST)
    elif name == "Hybrid Sequential":
        p = make_policy_Hybrid(agent, ACTIONS_LIST)
    
    agent_costs[name] = compute_cost(p, n_trails, n_steps, SPOT, STRIKE, MATURITY, R, VOL, INIT_POSITION, DT, MU, KAPPA)

### Results and graphs ###
print_hedge_table(agent_costs, OptionPrice)

if PLOT:
    plot_histogram(agent_costs)
    plot_learningcurve(rewards)
    plot_learningcurve_grid(rewards)
    plot_policy_3d(agents, ACTIONS_LIST, MATURITY, VOL)
    plot_hedge_trajectory(env, agents, ACTIONS_LIST, VOL)

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if REPORT: 
    build_report( agent_costs=agent_costs, rewards=rewards, OptionPrice=OptionPrice, 
                 selected_agents=agents, actions_list=ACTIONS_LIST, maturity=MATURITY, vol=VOL, env=env, output_path=f"hedging_report_{timestamp}.html")

print("Done!")