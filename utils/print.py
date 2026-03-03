
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_hedge_table(Cost_BSM, Cost_RL, OptionPrice):
    HedgeComp = pd.DataFrame(
        {
            "BSM": 100 * np.array([
                -np.mean(Cost_BSM),
                np.std(Cost_BSM)
            ]) / OptionPrice,

            "RL": 100 * np.array([
                -np.mean(Cost_RL),
                np.std(Cost_RL)
            ]) / OptionPrice
        },
        index=[
            "Average Hedge Cost (% of Option Price)",
            "STD Hedge Cost (% of Option Price)"
        ]
    )

    print(HedgeComp)

def plot_histogram(Cost_RL, Cost_BSM):
    num_bins = 10

    plt.figure(figsize=(10, 5))

    plt.hist(-Cost_RL, bins=num_bins, color='red', alpha=0.5, label='RL Hedge')
    plt.hist(-Cost_BSM, bins=num_bins, color='blue', alpha=0.5, label='Theoretical BLS Delta')

    plt.xlabel('Hedging Costs')
    plt.ylabel('Number of Trials')
    plt.title('RL Hedge Costs vs. BLS Hedge Costs')
    plt.legend(loc='best')



def plot_learningcurve(all_episode_rewards):
    plt.figure(figsize=(10, 5))

    # Raw rewards
    plt.plot(all_episode_rewards, label='Episode Reward', alpha=0.3, color='blue')

    # Calculate the rolling over 100 episode, to visualize trends
    rolling_mean = pd.Series(all_episode_rewards).rolling(window=100).mean()
    plt.plot(rolling_mean, label='Moving Average (100 episodes)', color='red', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DDPG Agent Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()