
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def print_hedge_table(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice):
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
            ]) / OptionPrice,

            "Hybrid": 100 * np.array([
                -np.mean(Cost_hybrid),
                np.std(Cost_hybrid)
            ]) / OptionPrice, 

            
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
    plt.hist(-Cost_TD3, bins=num_bins, color='black', alpha=0.5, label='TD3 hedge')
    plt.xlabel('Hedging Costs')
    plt.ylabel('Number of Trials')
    plt.title('TD3 Hedge Costs vs. BLS Hedge Costs')
    plt.legend(loc='best')



def plot_learningcurve(all_episode_rewards_DDPG, all_episode_rewards_DQN, all_episode_rewards_TD3):
    plt.figure(figsize=(10, 5))

    # Raw rewards (lite genomskinliga så de inte tar över)
    #plt.plot(all_episode_rewards_DDPG, label='Episode Reward DDPG', alpha=0.3, color='red')
    #plt.plot(all_episode_rewards_DQN, label='Episode Reward DQN', alpha=0.3, color='green')
    #plt.plot(all_episode_rewards_TD3, label='Episode Reward TD3', alpha=0.3, color='black')
    # Calculate the rolling men over 100 episodes to visualize trends
    rolling_mean_DDPG = pd.Series(all_episode_rewards_DDPG).rolling(window=100).mean()
    rolling_mean_DQN = pd.Series(all_episode_rewards_DQN).rolling(window=100).mean()
    rolling_mean_TD3 = pd.Series(all_episode_rewards_TD3).rolling(window=100).mean()

    plt.plot(rolling_mean_DDPG, label='Moving Average (100 episodes) DDPG', color='red', linewidth=2)
    plt.plot(rolling_mean_DQN, label='Moving Average (100 episodes) DQN', color='green', linewidth=2)
    plt.plot(rolling_mean_TD3, label='Moving Average (100 episodes) TD3', color='black', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agents Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    

    
def plot_learningcurve_DDPG(all_episode_rewards_DDPG):
    plt.figure(figsize=(10, 5))

    # Raw rewards (lite genomskinliga så de inte tar över)
    plt.plot(all_episode_rewards_DDPG, label='Episode Reward DDPG', alpha=0.3, color='red')
    # Calculate the rolling men over 100 episodes to visualize trends
    rolling_mean_DDPG = pd.Series(all_episode_rewards_DDPG).rolling(window=100).mean()
   
    plt.plot(rolling_mean_DDPG, label='Moving Average (100 episodes) DDPG', color='red', linewidth=2)

    plt.ylim(-180, 10)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DDPG Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    


def plot_learningcurve_DQN(all_episode_rewards_DQN):
    plt.figure(figsize=(10, 5))

    # Raw rewards (lite genomskinliga så de inte tar över)
    plt.plot(all_episode_rewards_DQN, label='Episode Reward DQN', alpha=0.3, color='green')

    rolling_mean_DQN = pd.Series(all_episode_rewards_DQN).rolling(window=100).mean()
    plt.plot(rolling_mean_DQN, label='Moving Average (100 episodes) DQN', color='green', linewidth=2)

    plt.ylim(-180, 10)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_learningcurve_hybrid(all_episode_rewards_hybrid): #, all_episode_rewards_DQN, all_episode_rewards_TD3):
    plt.figure(figsize=(10, 5))

    # Raw rewards (lite genomskinliga så de inte tar över)
    plt.plot(all_episode_rewards_hybrid, label='Episode Reward Hybrid', alpha=0.3, color='blue')
    #plt.plot(all_episode_rewards_DQN, label='Episode Reward DQN', alpha=0.3, color='green')
    #plt.plot(all_episode_rewards_TD3, label='Episode Reward TD3', alpha=0.3, color='black')
    # Calculate the rolling men over 100 episodes to visualize trends
    rolling_mean_DDPG = pd.Series(all_episode_rewards_hybrid).rolling(window=100).mean()
    #rolling_mean_DQN = pd.Series(all_episode_rewards_DQN).rolling(window=100).mean()
    #rolling_mean_TD3 = pd.Series(all_episode_rewards_TD3).rolling(window=100).mean()

    plt.plot(rolling_mean_DDPG, label='Moving Average (100 episodes) Hybrid', color='blue', linewidth=2)
    #plt.plot(rolling_mean_DQN, label='Moving Average (100 episodes) DQN', color='green', linewidth=2)
    #plt.plot(rolling_mean_TD3, label='Moving Average (100 episodes) TD3', color='black', linewidth=2)
    
    plt.ylim(-180, 10)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Hybrid Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    

def plot_learningcurve_TD3(all_episode_rewards_TD3):
    plt.figure(figsize=(10, 5))

    # Raw rewards (lite genomskinliga så de inte tar över)
    plt.plot(all_episode_rewards_TD3, label='Episode Reward TD3', alpha=0.3, color='black')
    # Calculate the rolling men over 100 episodes to visualize trends
    rolling_mean_TD3 = pd.Series(all_episode_rewards_TD3).rolling(window=100).mean()

    plt.plot(rolling_mean_TD3, label='Moving Average (100 episodes) TD3', color='black', linewidth=2)
    plt.ylim(-180, 10)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('TD3 Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


#väldigt oklart om detta fungerar... 
def plot_policy_heatmap(dqn_agent, td3_agent, actions_list, maturity, strike):
    """
    Skapar en heatmap över hybrid-policyens beslut.
    """
    moneyness_range = np.linspace(0.8, 1.2, 50)  # S/K från 0.8 till 1.2
    ttm_range = np.linspace(0, maturity, 50)       # Tid från 0 till maturity
    
    heatmap_data = np.zeros((len(moneyness_range), len(ttm_range)))

    dqn_agent.qnet.eval()
    td3_agent.actor.eval()

    for i, mR in enumerate(moneyness_range):
        for j, ttm in enumerate(ttm_range):
            # Antag en neutral startposition (t.ex. 0.5) för visualisering
            state = np.array([[mR, ttm, 0.5]])
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                # Hybrid-logik
                bin_idx = dqn_agent.qnet(state_tensor).argmax(dim=1).item()
                raw_td3 = td3_agent.actor(state_tensor).item()
                
                lower = actions_list[bin_idx]
                upper = actions_list[bin_idx + 1] if bin_idx + 1 < len(actions_list) else 1.0
                action = lower + (raw_td3 * (upper - lower))
                
                heatmap_data[i, j] = action

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=np.round(ttm_range, 3), 
                yticklabels=np.round(moneyness_range, 2), cmap="viridis")
    plt.title("Hybrid Policy Heatmap (Hedge Ratio)")
    plt.xlabel("Time to Maturity (TTM)")
    plt.ylabel("Moneyness (S/K)")
    plt.show()




def plot_hedge_trajectory(env, dqn_agent, td3_agent, actions_list, bs_delta):
    """
    Visar pris, agentens position och BS-delta under en enskild episod.
    """
    state = env.reset()
    done = False
    
    prices = [env.spot]
    agent_positions = [env.initPosition]
    bs_deltas = [bs_delta(env.spot, env.strike, env.rate, env.maturity, env.vol)]
    rewards = []

    while not done:
        state_tensor = torch.tensor([state], dtype=torch.float32)
        
        # Välj hybrid-action
        bin_idx = dqn_agent.qnet(state_tensor).argmax(dim=1).item()
        raw_td3 = td3_agent.actor(state_tensor).item()
        lower, upper = actions_list[bin_idx], (actions_list[bin_idx+1] if bin_idx+1 < len(actions_list) else 1.0)
        action = lower + (raw_td3 * (upper - lower))
        
        reward, next_state, done = env.step(action)
        
        prices.append(env.spot)
        agent_positions.append(action)
        bs_deltas.append(bs_delta(env.spot, env.strike, env.rate, env.maturity, env.vol))
        state = next_state

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Översta grafen: Prisutveckling
    ax1.plot(prices, label="Stock Price", color='black')
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.set_title("Hedge Trajectory Analysis")

    # Nedersta grafen: Position (Hedge)
    ax2.plot(agent_positions, label="Hybrid Agent Position", color='red', marker='o', markersize=4)
    ax2.plot(bs_deltas, label="Theoretical BS Delta", color='blue', linestyle='--')
    ax2.set_ylabel("Hedge Ratio")
    ax2.set_xlabel("Time Steps")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_hybrid_decomposition(dqn_agent, td3_agent, actions_list, num_samples=100):
    """
    Scatter plot som visar hur TD3 finjusterar DQN:s val.
    """
    # Skapa slumpmässiga states för test
    test_mR = np.random.uniform(0.9, 1.1, num_samples)
    test_TTM = np.random.uniform(0.01, 0.08, num_samples)
    test_Pos = np.random.uniform(0, 1, num_samples)
    
    dqn_bases = []
    final_actions = []

    for i in range(num_samples):
        state = torch.tensor([[test_mR[i], test_TTM[i], test_Pos[i]]], dtype=torch.float32)
        
        bin_idx = dqn_agent.qnet(state).argmax(dim=1).item()
        raw_td3 = td3_agent.actor(state).item()
        
        lower = actions_list[bin_idx]
        upper = actions_list[bin_idx + 1] if bin_idx + 1 < len(actions_list) else 1.0
        
        dqn_bases.append(lower) # Bas-valet
        final_actions.append(lower + (raw_td3 * (upper - lower)))

    plt.figure(figsize=(8, 6))
    plt.scatter(dqn_bases, final_actions, alpha=0.6, color='purple')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3) # 45-graders linje
    plt.title("DQN Base vs. Final Hybrid Action")
    plt.xlabel("DQN Selected Lower Bound")
    plt.ylabel("Final Action (Rescaled with TD3)")
    plt.grid(True, alpha=0.2)
    plt.show()
