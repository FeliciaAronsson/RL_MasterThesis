
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_hedge_table(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, OptionPrice):
    HedgeComp = pd.DataFrame(
        {
            "BSM": 100 * np.array([
                -np.mean(Cost_BSM),
                np.std(Cost_BSM)
            ]) / OptionPrice,

            "DDPG": 100 * np.array([
                -np.mean(Cost_DDPG),
                np.std(Cost_DDPG)
            ]) / OptionPrice,

            "DQN": 100 * np.array([
                -np.mean(Cost_DQN),
                np.std(Cost_DQN)
            ]) / OptionPrice, 

            "TD3": 100 * np.array([
                -np.mean(Cost_TD3),
                np.std(Cost_TD3)
            ]) / OptionPrice

            
        },
        index=[
            "Average Hedge Cost (% of Option Price)",
            "STD Hedge Cost (% of Option Price)"
        ]
    )

    print(HedgeComp)

def plot_histogram(Cost_DDPG, Cost_DQN, Cost_TD3, Cost_BSM):
    num_bins = 10

    plt.figure(figsize=(10, 5))

    plt.hist(-Cost_DDPG, bins=num_bins, color='red', alpha=0.5, label='RL(DDPG) Hedge')
    plt.hist(-Cost_BSM, bins=num_bins, color='blue', alpha=0.5, label='Theoretical BLS Delta')
    plt.hist(-Cost_DQN, bins=num_bins, color='green', alpha=0.5, label='DQN hedge')
    plt.hist(-Cost_TD3, bins=num_bins, color='blue', alpha=0.5, label='TD3 hedge')
    plt.xlabel('Hedging Costs')
    plt.ylabel('Number of Trials')
    plt.title('RL Hedge Costs vs. BLS Hedge Costs')
    plt.legend(loc='best')



def plot_learningcurve(all_episode_rewards_DDPG, all_episode_rewards_DQN, all_episode_rewards_TD3):
    plt.figure(figsize=(10, 5))

    # Raw rewards (lite genomskinliga så de inte tar över)
    plt.plot(all_episode_rewards_DDPG, label='Episode Reward DDPG', alpha=0.3, color='red')
    plt.plot(all_episode_rewards_DQN, label='Episode Reward DQN', alpha=0.3, color='green')
    plt.plot(all_episode_rewards_TD3, label='Episode Reward TD3', alpha=0.3, color='blue')
    # Calculate the rolling men over 100 episodes to visualize trends
    rolling_mean_DDPG = pd.Series(all_episode_rewards_DDPG).rolling(window=100).mean()
    rolling_mean_DQN = pd.Series(all_episode_rewards_DQN).rolling(window=100).mean()
    rolling_mean_TD3 = pd.Series(all_episode_rewards_TD3).rolling(window=100).mean()

    plt.plot(rolling_mean_DDPG, label='Moving Average (100 episodes) DDPG', color='red', linewidth=2)
    plt.plot(rolling_mean_DQN, label='Moving Average (100 episodes) DQN', color='green', linewidth=2)
    plt.plot(rolling_mean_TD3, label='Moving Average (100 episodes) TD3', color='blue', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agents Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()