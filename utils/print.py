import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import torch
from utils.bs import bs_delta

COLORS = {
    "BSM":    "#2196F3",
    "DDPG":   "#F44336",
    "DQN":    "#4CAF50",
    "TD3":    "#FF9800",
    "Hybrid": "#9C27B0",
}


def print_hedge_table(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice):
    """
    Summary table of hedging performance for all agents.
    Reports mean cost, std, mean/std ratio, and worst-case (5th percentile).
    All values expressed as percentage of option price.
    """
    def stats(cost):
        mean  = -np.mean(cost)
        std   =  np.std(cost)
        return 100 * np.array([mean, std]) / OptionPrice

    HedgeComp = pd.DataFrame(
        {
            "BSM":    stats(Cost_BSM),
            "DDPG":   stats(Cost_DDPG),
            "DQN":    stats(Cost_DQN),
            "TD3":    stats(Cost_TD3),
            "Hybrid": stats(Cost_hybrid),
        },
        index=[
            "Mean hedge cost (% option price)",
            "Std hedge cost  (% option price)",
        ]
    )
    print("\n" + "="*65)
    print("  Hedging Performance Summary")
    print("="*65)
    print(HedgeComp.round(3).to_string())
    print("="*65 + "\n")


def plot_histogram(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice):
    """
    Overlapping histogram of hedging cost distributions for all five strategies.
    Dashed vertical lines mark each distribution's mean.
    x-axis normalised by option price for scale-free comparison.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    costs = {
        "BSM":    Cost_BSM,
        "DDPG":   Cost_DDPG,
        "DQN":    Cost_DQN,
        "TD3":    Cost_TD3,
        "Hybrid": Cost_hybrid,
    }

    all_vals = np.concatenate([-c / OptionPrice * 100 for c in costs.values()])
    bins = np.linspace(
        np.percentile(all_vals, 1),
        np.percentile(all_vals, 99),
        40
    )

    for name, cost in costs.items():
        ax.hist(-cost / OptionPrice * 100, bins=bins,
                alpha=0.45, color=COLORS[name], label=name, edgecolor="none")
        ax.axvline(-np.mean(cost) / OptionPrice * 100,
                   color=COLORS[name], linewidth=1.8, linestyle="--")

    ax.set_xlabel("Hedging cost (% of option price)", fontsize=12)
    ax.set_ylabel("Number of trials", fontsize=12)
    ax.set_title("Distribution of hedging costs across 1,000 simulated paths", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig("plot_histogram.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_cost_bars(Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice):
    """
    Bar chart comparing mean hedging cost with +/- 1 std error bars.
    """
    agents = ["BSM", "DDPG", "DQN", "TD3", "Hybrid"]
    costs  = [Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid]

    means = [-np.mean(c) / OptionPrice * 100 for c in costs]
    stds  = [ np.std(c)  / OptionPrice * 100 for c in costs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(agents, means,
                  color=[COLORS[a] for a in agents],
                  alpha=0.8, edgecolor="white", linewidth=0.8, width=0.55)
    ax.errorbar(agents, means, yerr=stds,
                fmt="none", color="black",
                capsize=5, linewidth=1.5, capthick=1.5)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + 0.3,
                f"{mean:.1f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean hedging cost (% of option price)", fontsize=12)
    ax.set_title("Mean hedging cost +/- 1 standard deviation", fontsize=13)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("plot_cost_bars.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_learningcurve(rewards_DDPG, rewards_DQN, rewards_TD3, rewards_Hybrid,
                       window=100):
    """
    Rolling mean learning curves for all four agents on one plot.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    all_rewards = {
        "DDPG":   rewards_DDPG,
        "DQN":    rewards_DQN,
        "TD3":    rewards_TD3,
        "Hybrid": rewards_Hybrid,
    }

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


def plot_learningcurve_grid(rewards_DDPG, rewards_DQN, rewards_TD3, rewards_Hybrid,
                            window=100):
    """
    2x2 grid of individual learning curves, one panel per agent.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    fig.suptitle("Individual learning curves", fontsize=14)

    pairs = [
        ("DDPG",   rewards_DDPG,   axes[0, 0]),
        ("DQN",    rewards_DQN,    axes[0, 1]),
        ("TD3",    rewards_TD3,    axes[1, 0]),
        ("Hybrid", rewards_Hybrid, axes[1, 1]),
    ]

    for name, rewards, ax in pairs:
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


def plot_policy_heatmaps(ddpg_agent, dqn_agent, td3_agent, hybrid_agent,
                         actions_list, maturity, vol, n_grid=40):
    """
    2x3 grid of policy heatmaps showing each agent's hedge ratio as a function
    of moneyness (S/K) and time to maturity, with position fixed at 0.5.
    BSM delta shown as reference in top-left panel.
    """
    moneyness = np.linspace(0.8, 1.2, n_grid)
    ttm       = np.linspace(1e-4, maturity, n_grid)

    def compute_grid(policy_fn):
        grid = np.zeros((n_grid, n_grid))
        for i, mR in enumerate(moneyness):
            for j, t in enumerate(ttm):
                s = torch.tensor([[mR, t, 0.5]], dtype=torch.float32)
                with torch.no_grad():
                    grid[i, j] = policy_fn(s)
        return grid

    def ddpg_fn(s):
        ddpg_agent.actor.eval()
        return ddpg_agent.actor(s).item()

    def td3_fn(s):
        td3_agent.actor.eval()
        return td3_agent.actor(s).item()

    def dqn_fn(s):
        dqn_agent.qnet.eval()
        return actions_list[dqn_agent.qnet(s).argmax(dim=1).item()]

    def hybrid_fn(s):
        hybrid_agent.dqn.qnet.eval()
        hybrid_agent.td3.actor.eval()
        idx = hybrid_agent.dqn.qnet(s).argmax(dim=1).item()
        raw = hybrid_agent.td3.actor(s).item()
        lo  = actions_list[idx]
        hi  = actions_list[idx + 1] if idx + 1 < len(actions_list) else 1.0
      # Rescale TD3 output to [lower_bound, upper_bound]
        if raw < lo:
            action = lo - raw * (hi - lo)
            action = float(np.clip(action, 0.0, 1.0))
             
        elif raw == lo:
            action = lo
            action = float(np.clip(action, 0.0, 1.0))
             
        else:
            action = lo+ raw * (hi - lo)
            action = float(np.clip(action, 0.0, 1.0))

        return action
   

    bsm_grid = np.array([
        [bs_delta(mR, 1.0, 0.0, t, vol) for t in ttm]
        for mR in moneyness
    ])

    grids = [
        ("DDPG",   compute_grid(ddpg_fn)),
        ("DQN",    compute_grid(dqn_fn)),
        ("TD3",    compute_grid(td3_fn)),
        ("Hybrid", compute_grid(hybrid_fn)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(17, 9))
    fig.suptitle("Policy heatmaps - hedge ratio (moneyness vs TTM, position=0.5)",
                 fontsize=13)

    xt = [f"{t:.3f}" for t in ttm[::8]]
    yl = [f"{m:.2f}" for m in moneyness[::8]]

    def draw(ax, grid, title, color="black"):
        sns.heatmap(grid, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    cbar_kws={"label": "Hedge ratio"},
                    xticklabels=8, yticklabels=8)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Time to maturity")
        ax.set_ylabel("Moneyness (S/K)")
        ax.set_xticklabels(xt, rotation=45, fontsize=7)
        ax.set_yticklabels(yl, fontsize=7)

    draw(axes[0, 0], bsm_grid, "BSM (reference)", COLORS["BSM"])

    positions = [(0, 1), (0, 0), (1, 0), (1, 1)]
    for (r, c), (name, grid) in zip(positions, grids):
        draw(axes[r, c], grid, name, COLORS[name])

    #axes[1, 2].set_visible(False)
    plt.tight_layout()
    plt.savefig("plot_policy_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_policy_3d(ddpg_agent, dqn_agent, td3_agent, hybrid_agent,
                   actions_list, maturity, vol, n_grid=40):
    """
    3D Surface plots of policy (hedge ratio) vs Moneyness and TTM.
    """
    # Create the meshgrid
    # X: Time to Maturity, Y: Moneyness
    ttm = np.linspace(1e-4, maturity, n_grid)
    moneyness = np.linspace(0.8, 1.2, n_grid)
    T, M = np.meshgrid(ttm, moneyness)

    def compute_grid(policy_fn):
        grid = np.zeros((n_grid, n_grid))
        for i in range(n_grid):
            for j in range(n_grid):
                # Note: mR is from 'moneyness', t is from 'ttm'
                mR, t = moneyness[i], ttm[j]
                s = torch.tensor([[mR, t, 0.5]], dtype=torch.float32)
                with torch.no_grad():
                    grid[i, j] = policy_fn(s)
        return grid

    # Define policy functions (Logic remains same as your original)
    def ddpg_fn(s): return ddpg_agent.actor(s).item()
    def td3_fn(s): return td3_agent.actor(s).item()
    def dqn_fn(s): return actions_list[dqn_agent.qnet(s).argmax(dim=1).item()]
    
    def hybrid_fn(s):
        idx = hybrid_agent.dqn.qnet(s).argmax(dim=1).item()
        raw = hybrid_agent.td3.actor(s).item()
        lo, hi = actions_list[idx], actions_list[idx + 1] if idx + 1 < len(actions_list) else 1.0
        if raw < lo:
            action = lo - raw * (hi - lo)
            action = float(np.clip(action, 0.0, 1.0))
             
        elif raw == lo:
            action = lo
            action = float(np.clip(action, 0.0, 1.0))
             
        else:
            action = lo + raw * (hi - lo)
            action = float(np.clip(action, 0.0, 1.0))

        return action

    # Compute Grids
    bsm_grid = np.array([[bs_delta(mR, 1.0, 0.0, t, vol) for t in ttm] for mR in moneyness])
    
    grids = [
        ("BSM (Ref)", bsm_grid),
        ("DDPG",      compute_grid(ddpg_fn)),
        ("DQN",       compute_grid(dqn_fn)),
        ("TD3",       compute_grid(td3_fn)),
        ("Hybrid",    compute_grid(hybrid_fn)),
    ]

    # Setup Figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("3D Policy Surfaces: Hedge Ratio (Δ) vs Moneyness & TTM", fontsize=16)

    for i, (name, grid) in enumerate(grids):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(T, M, grid, cmap=cm.RdYlGn, 
                               linewidth=0, antialiased=True, alpha=0.8)
        
        # Formatting
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('TTM')
        ax.set_ylabel('Moneyness (S/K)')
        ax.set_zlabel('Hedge Ratio')
        ax.set_zlim(0, 1)
        
        # Adjusting the viewing angle for classic BSM perspective
        ax.view_init(elev=30, azim=-135)

    plt.tight_layout()
    plt.savefig("plot_policy_3d.png", dpi=150)
    plt.show()
    
def plot_hedge_trajectory(env, ddpg_agent, dqn_agent, td3_agent, hybrid_agent,
                          actions_list, vol):
    """
    Simulate one episode and plot each agent's hedge ratio alongside
    the BSM delta and the asset price.
    """
    np.random.seed(42)


    state = env.reset()
    done  = False

    prices  = [env.spot]
    bsm_pos = [bs_delta(env.spot, env.strike, env.rate, env.maturity, vol)]
    pos     = {name: [] for name in ["DDPG", "DQN", "TD3", "Hybrid"]}

    while not done:
        state = np.array(state)
        s = torch.tensor([state], dtype=torch.float32)
        

        with torch.no_grad():
            ddpg_agent.actor.eval()
            td3_agent.actor.eval()
            dqn_agent.qnet.eval()
            hybrid_agent.dqn.qnet.eval()
            hybrid_agent.td3.actor.eval()

            a_ddpg = ddpg_agent.actor(s).item()
            a_td3  = td3_agent.actor(s).item()
            a_dqn  = actions_list[dqn_agent.qnet(s).argmax(dim=1).item()]

            idx = hybrid_agent.dqn.qnet(s).argmax(dim=1).item()
            raw = hybrid_agent.td3.actor(s).item()
            lo  = actions_list[idx]
            hi  = actions_list[idx + 1] if idx + 1 < len(actions_list) else 1.0
            
            # a_hybrid = lo + raw * (hi - lo)

            # Rescale TD3 output to [lower_bound, upper_bound]
            if raw < lo:
                a_hybrid = lo - raw * (hi - lo)
                a_hybrid = float(np.clip(a_hybrid, 0.0, 1.0))

            elif raw == lo:
                a_hybrid = lo 
                a_hybrid = float(np.clip(a_hybrid, 0.0, 1.0))

            else:
                a_hybrid = lo + raw * (hi - lo)
                a_hybrid = float(np.clip(a_hybrid, 0.0, 1.0))

        pos["DDPG"].append(a_ddpg)
        pos["DQN"].append(a_dqn)
        pos["TD3"].append(a_td3)
        pos["Hybrid"].append(a_hybrid)

        reward, next_state, done = env.step(a_td3)


        prices.append(env.spot)
        bsm_pos.append(
            bs_delta(env.spot, env.strike, env.rate, env.maturity, vol))
        
        state = next_state
   

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle("Hedge trajectory - one simulated episode", fontsize=13)

    ax1.plot(range(len(prices)), prices, color="black", linewidth=1.5)
    ax1.set_ylabel("Asset price", fontsize=11)
    ax1.grid(True, alpha=0.25)

    ax2.plot(range(len(bsm_pos)), bsm_pos,
             color=COLORS["BSM"], linewidth=2, linestyle="--", label="BSM delta")
    for name, p in pos.items():
        ax2.plot(range(len(p)), p,
                 color=COLORS[name], linewidth=1.4, alpha=0.85, label=name)

    ax2.set_xlabel("Time step", fontsize=11)
    ax2.set_ylabel("Hedge ratio", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("plot_hedge_trajectory.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_hybrid_decomposition(hybrid_agent, actions_list, maturity,
                              num_samples=500):
    """
    Two-panel plot showing how TD3 fine-tunes DQN's coarse bin selection.
    Left: scatter of DQN base vs final action (colour = bin index).
    Right: histogram of the TD3 adjustment (final - DQN base).
    """
    np.random.seed(0)
    mR  = np.random.uniform(0.8, 1.2, num_samples)
    TTM = np.random.uniform(1e-4, maturity, num_samples)
    Pos = np.random.uniform(0, 1, num_samples)

    bases, finals, indices = [], [], []

    hybrid_agent.dqn.qnet.eval()
    hybrid_agent.td3.actor.eval()

    for i in range(num_samples):
        s = torch.tensor([[mR[i], TTM[i], Pos[i]]], dtype=torch.float32)
        with torch.no_grad():
            idx   = hybrid_agent.dqn.qnet(s).argmax(dim=1).item()
            raw   = hybrid_agent.td3.actor(s).item()
            lo    = actions_list[idx]
            hi    = actions_list[idx + 1] if idx + 1 < len(actions_list) else 1.0
            final = lo + raw * (hi - lo)
        bases.append(lo)
        finals.append(final)
        indices.append(idx)

    deltas = np.array(finals) - np.array(bases)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Hybrid agent decomposition", fontsize=13)

    sc = axes[0].scatter(bases, finals, c=indices, cmap="plasma",
                         alpha=0.5, s=18, edgecolors="none")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    plt.colorbar(sc, ax=axes[0], label="DQN bin index")
    axes[0].set_xlabel("DQN bin lower bound", fontsize=11)
    axes[0].set_ylabel("Final action (after TD3 fine-tuning)", fontsize=11)
    axes[0].set_title("DQN coarse selection vs. final action", fontsize=11)
    axes[0].grid(True, alpha=0.2)

    axes[1].hist(deltas, bins=30, color=COLORS["Hybrid"],
                 alpha=0.75, edgecolor="none")
    axes[1].axvline(0, color="black", linewidth=1.2, linestyle="--")
    axes[1].axvline(np.mean(deltas), color="red", linewidth=1.5,
                    label=f"Mean = {np.mean(deltas):.3f}")
    axes[1].set_xlabel("TD3 adjustment (final - DQN base)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Distribution of TD3 fine-tuning", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig("plot_hybrid_decomposition.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_learningcurve_DDPG(r):   _single_curve(r, "DDPG")
def plot_learningcurve_DQN(r):    _single_curve(r, "DQN")
def plot_learningcurve_TD3(r):    _single_curve(r, "TD3")
def plot_learningcurve_hybrid(r): _single_curve(r, "Hybrid")


def _single_curve(rewards, name, window=100):
    fig, ax = plt.subplots(figsize=(10, 4))
    s = pd.Series(rewards)
    ax.plot(s.values, alpha=0.2, color=COLORS[name], linewidth=0.8)
    ax.plot(s.rolling(window=window, min_periods=1).mean(),
            color=COLORS[name], linewidth=2,
            label=f"{window}-episode rolling mean")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Total reward", fontsize=11)
    ax.set_title(f"{name} learning curve", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"plot_learningcurve_{name.lower()}.png", dpi=150,
                bbox_inches="tight")
    plt.show()


def get_trajectory_data(env, dqn_agent, ddpg_agent, td3_agent, hybrid_agent, actions_list, vol):
    np.random.seed(42) # Samma seed som i din plot
    state = env.reset()
    done = False

    data = {
        "steps": [0],
        "prices": [env.spot], 
        "BSM": [bs_delta(env.spot, env.strike, env.rate, env.maturity, vol)],
        "DDPG": [], "DQN": [], "TD3": [], "Hybrid": []
    }

    while not done:
        s = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            # Sätt i eval-läge
            ddpg_agent.actor.eval(); td3_agent.actor.eval()
            dqn_agent.qnet.eval(); hybrid_agent.dqn.qnet.eval()
            hybrid_agent.td3.actor.eval()

            # Beräkna alla actions
            a_ddpg = ddpg_agent.actor(s).item()
            a_td3  = td3_agent.actor(s).item()
            a_dqn  = actions_list[dqn_agent.qnet(s).argmax(dim=1).item()]
            
            idx = hybrid_agent.dqn.qnet(s).argmax(dim=1).item()
            raw = hybrid_agent.td3.actor(s).item()
            lo, hi = actions_list[idx], (actions_list[idx+1] if idx+1 < len(actions_list) else 1.0)
            a_h = lo + raw * (hi - lo)

        # Spara positioner
        data["DDPG"].append(a_ddpg); data["DQN"].append(a_dqn)
        data["TD3"].append(a_td3); data["Hybrid"].append(a_h)

        # Stega framåt (vi följer TD3:s bana som i ditt exempel)
        _, next_state, done = env.step(a_td3)
        
        data["steps"].append(len(data["prices"]))
        data["prices"].append(env.spot)
        data["BSM"].append(bs_delta(env.spot, env.strike, env.rate, env.maturity, vol))
        state = next_state
    
    # Justera längden på de sista positionerna för att matcha pris-listan
    for name in ["DDPG", "DQN", "TD3", "Hybrid"]:
        data[name].append(data[name][-1]) # Håll sista positionen vid förfall
        
    return data

def get_policy_3d_grids(dqn_agent, ddpg_agent, td3_agent, hybrid_agent, actions_list, maturity, vol, n_grid=40):
    ttm_vec = np.linspace(1e-4, maturity, n_grid)
    mon_vec = np.linspace(0.8, 1.2, n_grid)
    
    # Initialize grids
    grids = {name: np.zeros((n_grid, n_grid)) for name in ["BSM", "DDPG", "DQN", "TD3", "Hybrid"]}

    # Ensure eval mode
    ddpg_agent.actor.eval(); td3_agent.actor.eval()
    dqn_agent.qnet.eval(); hybrid_agent.dqn.qnet.eval(); hybrid_agent.td3.actor.eval()

    with torch.no_grad():
        for i, m in enumerate(mon_vec):
            for j, t in enumerate(ttm_vec):
                s = torch.tensor([[m, t, 0.5]], dtype=torch.float32)
                
                # BSM
                grids["BSM"][i, j] = bs_delta(m, 1.0, 0.0, t, vol)
                
                # RL Agents
                grids["DDPG"][i, j] = ddpg_agent.actor(s).item()
                grids["TD3"][i, j]  = td3_agent.actor(s).item()
                grids["DQN"][i, j]  = actions_list[dqn_agent.qnet(s).argmax(dim=1).item()]
                
                # Hybrid Logic
                idx = hybrid_agent.dqn.qnet(s).argmax(dim=1).item()
                raw = hybrid_agent.td3.actor(s).item()
                lo = actions_list[idx]
                hi = actions_list[idx+1] if idx+1 < len(actions_list) else 1.0


                if raw < lo:
                    grids["Hybrid"][i, j] = lo - raw * (hi - lo)

                elif raw == lo:
                    grids["Hybrid"][i, j] = lo 
                
                else:
                    grids["Hybrid"][i, j] = lo + raw * (hi - lo)
                
    return grids