import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from utils.bs import bs_delta
from matplotlib import cm

COLORS = {
    "BSM":    "#2196F3",
    "DDPG":   "#F44336",
    "DQN":    "#4CAF50",
    "TD3":    "#FF9800",
    "Hybrid": "#9C27B0",
}


def print_hedge_table(agent_costs, OptionPrice):
    """
    Summary table of hedging performance for all agents.
    Reports mean cost, std, mean/std ratio, and worst-case (5th percentile).
    All values expressed as percentage of option price.
    """
    data = {}
    for name, cost in agent_costs.items():
        mean  = -np.mean(cost)
        std   =  np.std(cost)
        data[name] = 100 * np.array([mean, std]) / OptionPrice

    HedgeComp = pd.DataFrame(
        data,
        index = ["Mean hedge cost (% option price)", "Std hedge cost  (% option price)"]
    )
    print("\n" + "="*65)
    print("  Hedging Performance Summary ({',.join(agent_costs.keys())})")
    print("="*65)
    print(HedgeComp.round(3).to_string())
    print("="*65 + "\n")


def plot_histogram(agent_costs):
    """
    Overlapping histogram of hedging cost distributions for all five strategies.
    Dashed vertical lines mark each distribution's mean.
    x-axis normalised by option price for scale-free comparison.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    all_vals = np.concatenate([-c * 100 for c in agent_costs.values()])
    bins = np.linspace(
        np.percentile(all_vals, 1),
        np.percentile(all_vals, 99),
        40
    )

    for name, cost in agent_costs.items():
        ax.hist(-cost * 100, bins=bins,
                alpha=0.45, color=COLORS[name], label=name, edgecolor="none")
        ax.axvline(-np.mean(cost) * 100,
                   color=COLORS[name], linewidth=1.8, linestyle="--")

    ax.set_xlabel("Hedging cost", fontsize=12)
    ax.set_ylabel("Number of trials", fontsize=12)
    ax.set_title("Distribution of hedging costs across 1,000 simulated paths", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig("plot_histogram.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_learningcurve(all_rewards,
                       window=100):
    """
    Rolling mean learning curves for all four agents on one plot.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    for name, rewards in all_rewards.items():
        s = pd.Series(rewards)
        ax.plot(s.values, alpha=0.12, color=COLORS[name], linewidth=0.8)
        ax.plot(s.rolling(window=window, min_periods=1).mean(),
                label=f"{name} ({window}-ep avg)",
                color=COLORS[name], linewidth=2)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total episode reward", fontsize=12)
    ax.set_title("Learning curves - all agents", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("plot_learningcurve_all.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_learningcurve_grid(all_rewards,
                            window=100):
    """
    2x2 grid of individual learning curves, one panel per agent.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    fig.suptitle("Individual learning curves", fontsize=14)

    for name, rewards in all_rewards.items():
        ax = axes.flatten()[list(all_rewards.keys()).index(name)]
        s = pd.Series(rewards)
        ax.plot(s.values, alpha=0.2, color=COLORS[name], linewidth=0.8)
        ax.plot(s.rolling(window=window, min_periods=1).mean(),
                color=COLORS[name], linewidth=2,
                label=f"{window}-ep rolling mean")
        ax.set_title(name, fontsize=12, color=COLORS[name], fontweight="bold")
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Total reward", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("plot_learningcurve_grid.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_policy_3d(selected_agents, actions_list, maturity, vol, n_grid=25):
    """
    3D Surface plots of policy (hedge ratio) vs Moneyness and TTM.
    """
    ttm = np.linspace(1e-4, maturity, n_grid)
    moneyness = np.linspace(0.8, 1.2, n_grid)
    T, M = np.meshgrid(ttm, moneyness)

    def get_grid(agent_name, agent_obj):
        grid = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                mR, t = moneyness[i], ttm[j]
                s = torch.tensor([[mR, t, 0.5]], dtype=torch.float32)
                
                with torch.no_grad():
                    if agent_name == "DDPG" or agent_name == "TD3":
                        agent_obj.actor.eval()
                        val = agent_obj.actor(s).item()
                    elif agent_name == "DQN":
                        agent_obj.qnet.eval()
                        val = actions_list[agent_obj.qnet(s).argmax(dim=1).item()]
                    elif agent_name == "Hybrid":
                        agent_obj.dqn.qnet.eval()
                        agent_obj.td3.actor.eval()
                        idx = agent_obj.dqn.qnet(s).argmax(dim=1).item()
                        raw = agent_obj.td3.actor(s).item()
                        lo = actions_list[idx]
                        hi = actions_list[idx + 1] if idx + 1 < len(actions_list) else 1.0
                        if raw < lo:
                            val = np.clip(lo - raw * (hi - lo), 0.0, 1.0)
                        elif raw == lo:
                            val = lo
                        else:
                            val = np.clip(lo + raw * (hi - lo), 0.0, 1.0)
                    grid[i, j] = val
        return grid

    bsm_grid = np.array([[bs_delta(mR, 1.0, 0.0, t, vol) for t in ttm] for mR in moneyness])

    plot_list = [("BSM (Ref)", bsm_grid)]
    for name, agent in selected_agents.items():
        plot_list.append((name, get_grid(name, agent)))

    num_plots = len(plot_list)
    cols = 3
    rows = math.ceil(num_plots / cols)

    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.suptitle("3D Policy Surfaces: Hedge Ratio (Δ) vs Moneyness & TTM", fontsize=16)

    for i, (name, grid) in enumerate(plot_list):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(T, M, grid, cmap=cm.RdYlGn, 
                               linewidth=0, antialiased=True, alpha=0.8)
        
        color = COLORS.get(name.split()[0], "black")
        ax.set_title(name, fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('TTM')
        ax.set_ylabel('Moneyness (S/K)')
        ax.set_zlabel('Hedge Ratio')
        ax.set_zlim(0, 1)
        ax.view_init(elev=30, azim=-135)

    plt.tight_layout()
    plt.savefig("plot_policy_3d_dynamic.png", dpi=150)
    plt.show()

def plot_hedge_trajectory(env, selected_agents, actions_list, vol):
    state = env.reset()
    done = False

    prices = [env.spot]
    bsm_pos = [bs_delta(env.spot, env.strike, env.rate, env.maturity, vol)]
    
    agent_trajectories = {name: [] for name in selected_agents.keys()}

    while not done:
        s = torch.tensor([state], dtype=torch.float32)
        
        for name, agent in selected_agents.items():
            with torch.no_grad():
                if name == "DDPG" or name == "TD3":
                    val = agent.actor(s).item()
                elif name == "DQN":
                    val = actions_list[agent.qnet(s).argmax(dim=1).item()]
                elif name == "Hybrid":
                    idx = agent.dqn.qnet(s).argmax(dim=1).item()
                    raw = agent.td3.actor(s).item()
                    lo, hi = actions_list[idx], actions_list[idx+1] if idx+1 < len(actions_list) else 1.0
                    
                    if raw < lo:
                        val = float(np.clip(lo - raw * (hi - lo), 0.0, 1.0))    
                    elif raw == lo:
                        val = float(np.clip(lo, 0.0, 1.0))             
                    else:
                        val = float(np.clip(lo + raw * (hi - lo), 0.0, 1.0))
                agent_trajectories[name].append(val)

        first_agent_name = list(selected_agents.keys())[0]
        action_to_step = agent_trajectories[first_agent_name][-1]
        
        _, next_state, done = env.step(action_to_step)
        prices.append(env.spot)
        bsm_pos.append(bs_delta(env.spot, env.strike, env.rate, env.maturity, vol))
        state = next_state

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Hedge trajectories on one simulated price path", fontsize=13)
    ax1.plot(prices, color="black", label="Asset Price")
    ax1.legend()
    ax1.grid(True, alpha=0.25)
    ax1.set_ylabel("Asset Price")
    
    ax2.plot(bsm_pos, color=COLORS["BSM"], linestyle="--", label="BSM Delta")
    for name, trajectory in agent_trajectories.items():
        ax2.plot(trajectory, color=COLORS.get(name, "black"), label=name)
    
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Hedge ratio")
    ax2.grid(True, alpha=0.25)
    plt.show()