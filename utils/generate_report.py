import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

COLORS = {
    "BSM":    "#2196F3",
    "DDPG":   "#F44336",
    "DQN":    "#4CAF50",
    "TD3":    "#FF9800",
    "Hybrid": "#9C27B0",
}

def _get_policy_grid(name, agent, actions_list, moneyness, ttm):
    import torch
    n_m, n_t = len(moneyness), len(ttm)
    grid = np.zeros((n_m, n_t))
    for i, mR in enumerate(moneyness):
        for j, t in enumerate(ttm):
            s = torch.tensor([[mR, t, 0.5]], dtype=torch.float32)
            with torch.no_grad():
                if name in ("DDPG", "TD3"):
                    agent.actor.eval()
                    grid[i, j] = agent.actor(s).item()
                elif name == "DQN":
                    agent.qnet.eval()
                    grid[i, j] = actions_list[agent.qnet(s).argmax(dim=1).item()]
                elif name in ("Hybrid", "Hybrid Sequential"):
                    agent.dqn.qnet.eval()
                    agent.td3.actor.eval()
                    idx = agent.dqn.qnet(s).argmax(dim=1).item()
                    raw = agent.td3.actor(s).item()
                    lo  = actions_list[idx]
                    hi  = actions_list[idx + 1] if idx + 1 < len(actions_list) else 1.0
                    grid[i, j] = float(np.clip(lo + raw * (hi - lo), 0.0, 1.0))
    return grid


def _build_policy_3d_section(selected_agents, actions_list, maturity, vol,
                              n_grid=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import io, base64
    from utils.bs import bs_delta

    moneyness = np.linspace(0.8, 1.2, n_grid)
    ttm       = np.linspace(1e-4, maturity, n_grid)
    T, M      = np.meshgrid(ttm, moneyness)

    bsm_grid = np.array([
        [bs_delta(mR, 1.0, 0.0, t, vol) for t in ttm]
        for mR in moneyness
    ])

    plot_list = [("BSM", bsm_grid)]
    for name, agent in selected_agents.items():
        plot_list.append((name, _get_policy_grid(name, agent, actions_list,
                                                  moneyness, ttm)))

    n    = len(plot_list)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.suptitle("Policy surfaces: Hedge ratio vs Moneyness and TTM",
                 fontsize=15)

    for i, (name, grid) in enumerate(plot_list):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.plot_surface(T, M, grid, cmap=cm.RdYlGn,
                        linewidth=0, antialiased=True, alpha=0.85)
        color = COLORS.get(name.split()[0], "black")
        ax.set_title(name, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("TTM")
        ax.set_ylabel("Moneyness (S/K)")
        ax.set_zlabel("Hedge ratio")
        ax.set_zlim(0, 1)
        ax.view_init(elev=30, azim=-135)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px;" />'


def _build_hedge_trajectory_section(env, selected_agents, actions_list, vol):
    import torch
    from utils.bs import bs_delta

    np.random.seed(42)
    state = env.reset()
    done  = False

    prices       = [env.spot]
    bsm_pos      = [bs_delta(env.spot, env.strike, env.rate, env.maturity, vol)]
    trajectories = {name: [] for name in selected_agents}

    while not done:
        s = torch.tensor([state], dtype=torch.float32)
        for name, agent in selected_agents.items():
            with torch.no_grad():
                if name in ("DDPG", "TD3"):
                    agent.actor.eval()
                    val = agent.actor(s).item()
                elif name == "DQN":
                    agent.qnet.eval()
                    val = actions_list[agent.qnet(s).argmax(dim=1).item()]
                elif name in ("Hybrid", "Hybrid Sequential"):
                    agent.dqn.qnet.eval()
                    agent.td3.actor.eval()
                    idx = agent.dqn.qnet(s).argmax(dim=1).item()
                    raw = agent.td3.actor(s).item()
                    lo  = actions_list[idx]
                    hi  = actions_list[idx + 1] if idx + 1 < len(actions_list) else 1.0
                    val = float(np.clip(lo + raw * (hi - lo), 0.0, 1.0))
            trajectories[name].append(val)

        first_action = trajectories[list(selected_agents.keys())[0]][-1]
        _, next_state, done = env.step(first_action)
        prices.append(env.spot)
        bsm_pos.append(
            bs_delta(env.spot, env.strike, env.rate, env.maturity, vol))
        state = next_state

    steps = list(range(len(prices)))
    fig   = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Asset price", "Hedge ratio"),
        vertical_spacing=0.1, row_heights=[0.3, 0.7],
    )
    fig.add_trace(go.Scatter(
        x=steps, y=prices, mode="lines",
        line=dict(color="black", width=1.5),
        name="Asset price", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(range(len(bsm_pos))), y=bsm_pos, mode="lines",
        line=dict(color=COLORS["BSM"], width=2, dash="dash"),
        name="BSM delta",
    ), row=2, col=1)
    for name, traj in trajectories.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(traj))), y=traj, mode="lines",
            line=dict(color=COLORS.get(name, "#888"), width=1.8),
            name=name,
        ), row=2, col=1)

    fig.update_layout(
        height=560, hovermode="x unified",
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=50, b=60),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#eee", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#eee",
                     range=[-0.05, 1.05], row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#eee",
                     title_text="Time step", row=2, col=1)
    return fig


def build_report(
    agent_costs,
    rewards,
    OptionPrice,
    output_path="hedging_report.html",
    selected_agents=None,
    actions_list=None,
    maturity=None,
    vol=None,
    env=None,
):
    """
    Generate an HTML report with hedging results.

    :param agent_costs:     dict {name: cost_array} — include "BSM" for benchmark.
    :param rewards:         dict {name: reward_list} — only RL agents.
    :param OptionPrice:     option price used for normalisation.
    :param selected_agents: dict {name: agent_object} for sections 6 and 7.
    :param actions_list:    discrete action grid (from config.ACTIONS_LIST).
    :param maturity:        option maturity in years (from config.MATURITY).
    :param vol:             volatility (from config.VOL).
    :param env:             HedgingEnv instance (for trajectory plot).

    Example — matches main.py structure exactly:
        build_report(
            agent_costs=agent_costs,
            rewards=rewards,
            OptionPrice=OptionPrice,
            selected_agents=agents,
            actions_list=ACTIONS_LIST,
            maturity=MATURITY,
            vol=VOL,
            env=env,
        )
    """
    costs       = {a: np.array(c, dtype=float) for a, c in agent_costs.items()}
    cost_agents = list(costs.keys())
    rl_agents   = [a for a in cost_agents if a != "BSM" and a in rewards]

    means = {a: -np.mean(c) / OptionPrice * 100 for a, c in costs.items()}
    stds  = {a:  np.std(c)  / OptionPrice * 100 for a, c in costs.items()}

    sections = []

    # HTML header
    sections.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hedging using Deep Reinforcement Learning - Results</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; background: #f8f9fa; color: #1a1a1a; }
  .header { background: #1a1a2e; color: white; padding: 48px 60px 36px; }
  .header h1 { margin: 0 0 8px; font-size: 28px; font-weight: 600; }
  .header p  { margin: 0; font-size: 15px; color: #aab; }
  .container { max-width: 1200px; margin: 0 auto; padding: 40px; }
  .section { background: white; border-radius: 12px; padding: 32px 36px;
             margin-bottom: 32px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .section h2 { margin: 0 0 6px; font-size: 20px; font-weight: 600; color: #1a1a2e; }
  .section .desc { font-size: 14px; color: #555; margin: 0 0 20px;
                   line-height: 1.6; max-width: 760px; }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  th { background: #1a1a2e; color: white; padding: 10px 16px;
       text-align: left; font-weight: 500; }
  td { padding: 9px 16px; border-bottom: 1px solid #eee; }
  tr:last-child td { border-bottom: none; }
  tr:nth-child(even) td { background: #f8f9fa; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 20px;
           font-size: 12px; font-weight: 500; }
</style>
</head>
<body>
<div class="header">
  <h1>Hedging using Deep Reinforcement Learning</h1>
  <p>Master thesis, Felicia Aronsson &amp; Jelena N&auml;&auml;s, Ume&aring; University, 2026</p>
</div>
<div class="container">
""")

    # 1. Summary table
    header_cols = "".join(
        f'<th><span class="badge" style="background:{COLORS[a]}22;color:{COLORS[a]}">{a}</span></th>'
        for a in cost_agents
    )
    mean_cols = "".join(f"<td>{means[a]:.3f}</td>" for a in cost_agents)
    std_cols  = "".join(f"<td>{stds[a]:.3f}</td>"  for a in cost_agents)

    sections.append(f"""
<div class="section">
  <h2>1. Performance summary</h2>
  <p class="desc">All hedging costs are expressed as a percentage of the option price.</p>
  <table>
    <thead>
      <tr><th></th>{header_cols}</tr>
    </thead>
    <tbody>
      <tr><td><strong>Mean cost (%)</strong></td>{mean_cols}</tr>
      <tr><td><strong>Std (%)</strong></td>{std_cols}</tr>
    </tbody>
  </table>
</div>
""")

    # 2. Histograms — each RL agent vs BSM
    rl_with_cost = [(a, costs[a]) for a in rl_agents if a in costs]
    if rl_with_cost and "BSM" in costs:
        n = len(rl_with_cost)
        cols = min(2, n)
        rows = math.ceil(n / cols)

        fig_hist = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{name} vs BSM" for name, _ in rl_with_cost],
            horizontal_spacing=0.12, vertical_spacing=0.2
        )

        bsm_pct = (-costs["BSM"] / OptionPrice * 100).tolist()

        for idx, (name, cost) in enumerate(rl_with_cost):
            row = idx // cols + 1
            col = idx %  cols + 1
            show_legend = (idx == 0)
            rl_pct = (-cost / OptionPrice * 100).tolist()

            fig_hist.add_trace(go.Histogram(
                x=bsm_pct, nbinsx=30, name="BSM",
                marker_color=COLORS["BSM"], opacity=0.55,
                showlegend=show_legend, legendgroup="BSM",
            ), row=row, col=col)

            fig_hist.add_trace(go.Histogram(
                x=rl_pct, nbinsx=30, name=name,
                marker_color=COLORS[name], opacity=0.65,
                showlegend=show_legend, legendgroup=name,
            ), row=row, col=col)

            for val, color in [
                (float(np.mean(bsm_pct)), COLORS["BSM"]),
                (float(np.mean(rl_pct)),  COLORS[name])
            ]:
                fig_hist.add_trace(go.Scatter(
                    x=[val, val], y=[0, 200], mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ), row=row, col=col)

        fig_hist.update_layout(
            barmode="overlay",
            height=320 * rows,
            legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
            margin=dict(l=20, r=20, t=50, b=60),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig_hist.update_xaxes(showgrid=True, gridcolor="#eee",
                              title_text="Hedging cost (% of option price)")
        fig_hist.update_yaxes(showgrid=True, gridcolor="#eee", title_text="Count")

        sections.append(f"""
<div class="section">
  <h2>2. Hedging cost distributions</h2>
  <p class="desc">Each panel compares one RL agent against the BSM benchmark.
  Dashed vertical lines mark the mean of each distribution.
  A distribution shifted to the right indicates lower (better) average hedging costs.</p>
  {fig_hist.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")

    # 3. Bar chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=cost_agents,
        y=[means[a] for a in cost_agents],
        error_y=dict(type="data",
                     array=[stds[a] for a in cost_agents],
                     visible=True, color="rgba(0,0,0,0.5)",
                     thickness=2, width=6),
        marker_color=[COLORS[a] for a in cost_agents],
        marker_line_color="white", marker_line_width=1.5,
        opacity=0.85,
        text=[f"{means[a]:.2f}" for a in cost_agents],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=420,
        yaxis_title="Mean hedging cost (% of option price)",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    fig_bar.update_yaxes(showgrid=True, gridcolor="#eee")

    sections.append(f"""
<div class="section">
  <h2>3. Mean hedging cost with standard deviation</h2>
  <p class="desc">Error bars show plus/minus 1 standard deviation.
  A lower bar means cheaper hedging on average; shorter error bars mean
  more consistent performance.</p>
  {fig_bar.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")

    # 4. Combined learning curves
    if rl_agents:
        window = 100
        fig_lc = go.Figure()
        for name in rl_agents:
            s = pd.Series([float(r) for r in rewards[name]])
            rolling = s.rolling(window=window, min_periods=1).mean()
            fig_lc.add_trace(go.Scatter(
                y=s.values.tolist(), mode="lines",
                line=dict(color=COLORS[name], width=0.8),
                opacity=0.15, showlegend=False, hoverinfo="skip",
            ))
            fig_lc.add_trace(go.Scatter(
                y=rolling.values.tolist(), mode="lines",
                name=f"{name} ({window}-ep avg)",
                line=dict(color=COLORS[name], width=2.2),
            ))
        fig_lc.update_layout(
            height=440, xaxis_title="Episode",
            yaxis_title="Total episode reward",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=20, r=20, t=20, b=60),
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
            hovermode="x unified",
        )
        fig_lc.update_xaxes(showgrid=True, gridcolor="#eee")
        fig_lc.update_yaxes(showgrid=True, gridcolor="#eee")

        sections.append(f"""
<div class="section">
  <h2>4. Learning curves</h2>
  <p class="desc">Raw episode rewards shown at low opacity; solid lines are
  {window}-episode rolling means.</p>
  {fig_lc.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")

    # 5. Individual learning curves grid
    if len(rl_agents) > 1:
        cols_ind = min(2, len(rl_agents))
        rows_ind = math.ceil(len(rl_agents) / cols_ind)

        fig_ind = make_subplots(
            rows=rows_ind, cols=cols_ind,
            subplot_titles=rl_agents,
            horizontal_spacing=0.1, vertical_spacing=0.2
        )
        for idx, name in enumerate(rl_agents):
            row = idx // cols_ind + 1
            col = idx %  cols_ind + 1
            s = pd.Series([float(r) for r in rewards[name]])
            rolling = s.rolling(window=window, min_periods=1).mean()

            fig_ind.add_trace(go.Scatter(
                y=s.values.tolist(), mode="lines",
                line=dict(color=COLORS[name], width=0.8),
                opacity=0.2, showlegend=False, hoverinfo="skip",
            ), row=row, col=col)

            fig_ind.add_trace(go.Scatter(
                y=rolling.values.tolist(), mode="lines",
                name=name, line=dict(color=COLORS[name], width=2),
                showlegend=False,
            ), row=row, col=col)

        fig_ind.update_layout(
            height=300 * rows_ind,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        fig_ind.update_xaxes(showgrid=True, gridcolor="#eee", title_text="Episode")
        fig_ind.update_yaxes(showgrid=True, gridcolor="#eee", title_text="Reward")

        sections.append(f"""
<div class="section">
  <h2>5. Individual learning curves</h2>
  <p class="desc">Each panel shows one agent's training progress in detail.</p>
  {fig_ind.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")

    # 6. Policy 3D surfaces
    if selected_agents and actions_list is not None and maturity is not None and vol is not None:
        img_tag = _build_policy_3d_section(selected_agents, actions_list,
                                           maturity, vol)
        sections.append(f"""
<div class="section">
  <h2>6. Policy surfaces</h2>
  <p class="desc">3D surface plots showing each agent's hedge ratio as a function
  of moneyness (S/K) and time to maturity, with position fixed at 0.5.
  BSM delta is shown as reference.</p>
  {img_tag}
</div>
""")

    # 7. Hedge trajectory
    if selected_agents and env is not None and actions_list is not None and vol is not None:
        fig_traj = _build_hedge_trajectory_section(env, selected_agents,
                                                    actions_list, vol)
        sections.append(f"""
<div class="section">
  <h2>7. Hedge trajectory</h2>
  <p class="desc">One simulated episode showing how each agent's hedge ratio
  evolves over time alongside the asset price and the BSM delta benchmark.
  Hover over the chart for exact values at each time step.</p>
  {fig_traj.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")

    # Footer
    sections.append("""
</div>
<div style="text-align:center;padding:24px;font-size:12px;color:#999;">
  Generated automatically from trained agents &mdash; RL_MasterThesis
</div>
</body>
</html>
""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(sections))

    print(f"Report saved to: {output_path}")