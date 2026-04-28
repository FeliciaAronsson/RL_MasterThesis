import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from utils.bs import bs_delta
COLORS = {
    "BSM":    "#2196F3",
    "DDPG":   "#F44336",
    "DQN":    "#4CAF50",
    "TD3":    "#FF9800",
    "Hybrid": "#9C27B0",
}

def build_report(
    Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice,
    rewards_DDPG, rewards_DQN, rewards_TD3, rewards_Hybrid, trajectory_data, policy_data, maturity,
    output_path="hedging_report.html"
):
    Cost_BSM    = np.array(Cost_BSM,    dtype=float)
    Cost_DDPG   = np.array(Cost_DDPG,   dtype=float)
    Cost_DQN    = np.array(Cost_DQN,    dtype=float)
    Cost_TD3    = np.array(Cost_TD3,    dtype=float)
    Cost_hybrid = np.array(Cost_hybrid, dtype=float)
    rewards_DDPG   = list(rewards_DDPG)
    rewards_DQN    = list(rewards_DQN)
    rewards_TD3    = list(rewards_TD3)
    rewards_Hybrid = list(rewards_Hybrid)

    agents = ["BSM", "DDPG", "DQN", "TD3", "Hybrid"]
    costs  = [Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid]

    means  = [-np.mean(c) / OptionPrice * 100 for c in costs]
    stds   = [ np.std(c)  / OptionPrice * 100 for c in costs]

    sections = []

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
  <p>Master thesis work by Felicia Aronsson &amp; Jelena N&auml;&auml;s, Ume&aring; University, 2026</p>
</div>
<div class="container">
""")
    
# ── 1. Summary table ──────────────────────────────────────────────────────
    header_cols = "".join(
        f'<th><span class="badge" style="background:{COLORS[a]}22;color:{COLORS[a]}">{a}</span></th>'
        for a in agents
    )
    mean_cols = "".join(f"<td>{m:.3f}</td>" for m in means)
    std_cols  = "".join(f"<td>{s:.3f}</td>" for s in stds)

    sections.append(f"""
<div class="section">
  <h2>1. Performance summary</h2>
  <p class="desc">All hedging costs are expressed as a percentage of the option price.</p>
  <table>
    <thead>
      <tr>
        <th></th>
        {header_cols}
      </tr>
    </thead>
    <tbody>
      <tr><td><strong>Mean cost (%)</strong></td>{mean_cols}</tr>
      <tr><td><strong>Std (%)</strong></td>{std_cols}</tr>
    </tbody>
  </table>
</div>
""")

    # ── 2. Histograms 2x2 — each RL agent vs BSM ─────────────────────────────
    rl_agents = [("DDPG", Cost_DDPG), ("DQN", Cost_DQN),
                 ("TD3", Cost_TD3),   ("Hybrid", Cost_hybrid)]

    fig_hist = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{name} vs BSM" for name, _ in rl_agents],
        horizontal_spacing=0.12, vertical_spacing=0.2
    )

    bsm_vals_pct = (-Cost_BSM / OptionPrice * 100).tolist()

    for idx, (name, cost) in enumerate(rl_agents):
        row = idx // 2 + 1
        col = idx %  2 + 1
        show_legend = (idx == 0)

        rl_vals_pct = (-cost / OptionPrice * 100).tolist()

        fig_hist.add_trace(go.Histogram(
            x=bsm_vals_pct,
            nbinsx=30,
            name="BSM",
            marker_color=COLORS["BSM"],
            opacity=0.55,
            showlegend=show_legend,
            legendgroup="BSM",
        ), row=row, col=col)

        fig_hist.add_trace(go.Histogram(
            x=rl_vals_pct,
            nbinsx=30,
            name=name,
            marker_color=COLORS[name],
            opacity=0.65,
            showlegend=show_legend,
            legendgroup=name,
        ), row=row, col=col)

        bsm_mean = float(np.mean(bsm_vals_pct))
        rl_mean  = float(np.mean(rl_vals_pct))

        for val, color in [(bsm_mean, COLORS["BSM"]), (rl_mean, COLORS[name])]:
            fig_hist.add_trace(go.Scatter(
                x=[val, val],
                y=[0, 200],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ), row=row, col=col)

    fig_hist.update_layout(
        barmode="overlay",
        height=580,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=50, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
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

    # ── 3. Bar chart ──────────────────────────────────────────────────────────
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=agents,
        y=means,
        error_y=dict(type="data", array=stds, visible=True,
                     color="rgba(0,0,0,0.5)", thickness=2, width=6),
        marker_color=[COLORS[a] for a in agents],
        marker_line_color="white",
        marker_line_width=1.5,
        opacity=0.85,
        text=[f"{m:.2f}" for m in means],
        textposition="outside",
    ))
    fig_bar.update_layout(
        height=420,
        yaxis_title="Mean hedging cost (% of option price)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    fig_bar.update_yaxes(showgrid=True, gridcolor="#eee")

    sections.append(f"""
<div class="section">
  <h2>3. Mean hedging cost with standard deviation</h2>
  <p class="desc">Error bars show plus/minus 1 standard deviation.
  A lower bar means cheaper hedging on average; shorter error bars mean
  more consistent performance across simulated paths.</p>
  {fig_bar.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")

    # ── 4. Combined learning curves ───────────────────────────────────────────
    window = 100
    all_lr = {
        "DDPG":   rewards_DDPG,
        "DQN":    rewards_DQN,
        "TD3":    rewards_TD3,
        "Hybrid": rewards_Hybrid,
    }

    fig_lc = go.Figure()
    for name, rewards in all_lr.items():
        s = pd.Series([float(r) for r in rewards])
        rolling = s.rolling(window=window, min_periods=1).mean()

        fig_lc.add_trace(go.Scatter(
            y=s.values.tolist(),
            mode="lines",
            line=dict(color=COLORS[name], width=0.8),
            opacity=0.15,
            showlegend=False,
            hoverinfo="skip",
        ))
        fig_lc.add_trace(go.Scatter(
            y=rolling.values.tolist(),
            mode="lines",
            name=f"{name} ({window}-ep avg)",
            line=dict(color=COLORS[name], width=2.2),
        ))

    fig_lc.update_layout(
        height=440,
        xaxis_title="Episode",
        yaxis_title="Total episode reward",
        plot_bgcolor="white",
        paper_bgcolor="white",
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
  {window}-episode rolling means. An upward trend indicates improving policy quality.
  Hover over the chart for exact values.</p>
  {fig_lc.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")

    # ── 5. Individual learning curves 2x2 ────────────────────────────────────
    fig_ind = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(all_lr.keys()),
        horizontal_spacing=0.1, vertical_spacing=0.2
    )
    for idx, (name, rewards) in enumerate(all_lr.items()):
        row = idx // 2 + 1
        col = idx %  2 + 1
        s = pd.Series([float(r) for r in rewards])
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
        height=560,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig_ind.update_xaxes(showgrid=True, gridcolor="#eee", title_text="Episode")
    fig_ind.update_yaxes(showgrid=True, gridcolor="#eee", title_text="Reward")

    sections.append(f"""
<div class="section">
  <h2>5. Individual learning curves</h2>
  <p class="desc">Each panel shows one agent's training progress in detail,
  making it easier to inspect convergence speed and training stability.</p>
  {fig_ind.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")
# ── 6. Hedge Trajectory (Dynamic) ────────────────────────────────────────

    if trajectory_data:
        
        fig_traj = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.3, 0.7],
            subplot_titles=("Asset Price Path", "Agent Hedge Positions (Delta)")
        )

        # Rad 1: Asset Price
        fig_traj.add_trace(go.Scatter(
            x=trajectory_data["steps"], y=trajectory_data["prices"],
            name="Asset Price", line=dict(color="black", width=2)
        ), row=1, col=1)

        # Rad 2: BSM Delta (dashed)
        fig_traj.add_trace(go.Scatter(
            x=trajectory_data["steps"], y=trajectory_data["BSM"],
            name="BSM Delta", line=dict(color=COLORS["BSM"], width=2, dash='dash')
        ), row=2, col=1)

        # Rad 2: RL Agents
        for name in ["DDPG", "DQN", "TD3", "Hybrid"]:
            fig_traj.add_trace(go.Scatter(
                x=trajectory_data["steps"], y=trajectory_data[name],
                name=name, line=dict(color=COLORS[name], width=1.5),
                opacity=0.8
            ), row=2, col=1)

        fig_traj.update_layout(
            height=700,
            plot_bgcolor="white",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")
        )
        fig_traj.update_yaxes(showgrid=True, gridcolor="#eee")
        fig_traj.update_xaxes(showgrid=True, gridcolor="#eee")

        sections.append(f"""
<div class="section">
    <h2>6. Detailed Hedge Trajectory</h2>
    <p class="desc">A step-by-step look at one simulated episode. 
    The upper panel shows the price evolution, and the lower panel shows 
    how each agent adjusted its hedge ratio in response.</p>
    {fig_traj.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")
        
 # ── 7. 3D Policy Surface Analysis ────────────────────────────────────────
    if policy_data:
        n_grid = 40
        ttm_vec = np.linspace(1e-4, maturity, n_grid)
        mon_vec = np.linspace(0.8, 1.2, n_grid)
        fig_3d = go.Figure()

        fig_3d = make_subplots(
            rows=2, cols=3, 
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.3, 0.7],
            subplot_titles=("Asset Price Path", "Agent Hedge Positions (Delta)")
        )
        # Add BSM as a semi-transparent reference surface that is always visible
        fig_3d.add_trace(go.Surface(
            x = ttm_vec, y = mon_vec, z=policy_data["BSM"],
            colorscale='Blues', opacity=0.4, name="BSM (Ref)", showscale=False
        ))

        # Add traces for each RL agent (initially hidden except the first one)
        for name in ["DDPG", "DQN", "TD3", "Hybrid"]:
            fig_3d.add_trace(go.Surface(
                x=ttm_vec, y=mon_vec, z=policy_data[name],
                colorscale='RdYlGn', name=name, visible=False, showscale=True,
                colorbar=dict(title="Hedge Ratio", x=1.1)
            ))

        # Make the first RL agent (DDPG) visible by default
        fig_3d.data[1].visible = True

        fig_3d.update_layout(
                scene=dict(
                xaxis_title='Time to Maturity',
                yaxis_title='Moneyness (S/K)',
                zaxis_title='Hedge Ratio (Δ)',
                camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2))
            ),
            height=800,
            margin=dict(t=100, b=50, l=50, r=50)
        )

    sections.append(f"""
<div class="section">
    <h2>7. Policy Surface Topography</h2>
    <p class="desc">This 3D surface shows the agent's "decision map." The ghosted blue surface is the 
    Black-Scholes Delta. Use the dropdown to compare different RL architectures.</p>
    {fig_3d.to_html(full_html=False, include_plotlyjs=False)}
</div>
""")
#
    # ── Footer ────────────────────────────────────────────────────────────────
    sections.append("""
</div>
<div style="text-align:center;padding:24px;font-size:12px;color:#999;">
  Generated automatically from trained agents &mdash; RL_MasterThesis @ hybridAgent
</div>
</body>
</html>
""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(sections))

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    np.random.seed(0)
    n = 1000
    ep = 500

    Cost_BSM    = np.random.normal(-2.1, 1.2, n)
    Cost_DDPG   = np.random.normal(-2.6, 1.8, n)
    Cost_DQN    = np.random.normal(-2.4, 2.0, n)
    Cost_TD3    = np.random.normal(-2.2, 1.5, n)
    Cost_hybrid = np.random.normal(-2.0, 1.3, n)
    OptionPrice = 3.0

    def fake(start, end, noise=8):
        return (np.linspace(start, end, ep) +
                np.random.normal(0, noise, ep)).tolist()

    build_report(
        Cost_BSM, Cost_DDPG, Cost_DQN, Cost_TD3, Cost_hybrid, OptionPrice,
        fake(-60, -15), fake(-55, -18), fake(-50, -12), fake(-52, -14),
        output_path="/mnt/user-data/outputs/hedging_report.html"
    )