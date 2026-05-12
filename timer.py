"""
=============================================================================
scripts/generate_demo_plots.py  –  Offline Demo Visualization Generator
=============================================================================
Generates all showcase plots using synthetic data — no GPU, no training needed.
Produces a complete visual portfolio of the system's capabilities.

Usage:
    python scripts/generate_demo_plots.py
=============================================================================
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force Matplotlib non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

OUT_DIR = Path("evaluation/demo_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Aerospace dark theme ──────────────────────────────────────────────────────
DARK = {
    "figure.facecolor":  "#08111f",
    "axes.facecolor":    "#0d1b2e",
    "axes.edgecolor":    "#1e3050",
    "axes.labelcolor":   "#a0c0e0",
    "text.color":        "#a0c0e0",
    "xtick.color":       "#607090",
    "ytick.color":       "#607090",
    "grid.color":        "#162030",
    "grid.alpha":        1.0,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

ACCENT  = "#00d4ff"
GREEN   = "#00ff88"
YELLOW  = "#ffd700"
ORANGE  = "#ff6b35"
PURPLE  = "#cc44ff"
RED     = "#ff3355"


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Data Generators
# ──────────────────────────────────────────────────────────────────────────────

def gen_trajectory(n=1500, env="urban"):
    """Simulate a realistic flight trajectory with waypoints and avoidance."""
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 60, n)
    dt  = t[1] - t[0]

    # Waypoints (spiral + avoidance bumps)
    wp_x = [10, 50, 120, 160, 130, 80, 180]
    wp_y = [10, 40,  20,  80, 150, 180, 170]
    wp_z = [15, 20,  18,  25,  20,  22,  18]

    # Interpolate smooth path through waypoints
    from numpy.polynomial import polynomial as P
    ts   = np.linspace(0, 1, len(wp_x))
    ts_q = np.linspace(0, 1, n)
    xs = np.interp(ts_q, ts, wp_x) + rng.normal(0, 0.8, n)
    ys = np.interp(ts_q, ts, wp_y) + rng.normal(0, 0.8, n)
    zs = np.interp(ts_q, ts, wp_z) + rng.normal(0, 0.3, n)

    # Smooth
    from numpy import convolve
    k   = np.ones(30) / 30
    xs  = convolve(xs, k, mode="same")
    ys  = convolve(ys, k, mode="same")
    zs  = convolve(zs, k, mode="same")
    zs  = np.clip(zs, 5, 50)

    # Speed
    vx    = np.gradient(xs, dt)
    vy    = np.gradient(ys, dt)
    vz    = np.gradient(zs, dt)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    battery = 100 - np.linspace(0, 28, n) + rng.normal(0, 0.2, n)
    gnss_q  = 0.95 - 0.05 * np.sin(ts_q * 20) + rng.normal(0, 0.02, n)
    gnss_q  = np.clip(gnss_q, 0.2, 1.0)
    reward  = np.cumsum(rng.normal(0.3, 1.5, n))

    return dict(t=t, xs=xs, ys=ys, zs=zs, speed=speed,
                vx=vx, vy=vy, vz=vz,
                battery=battery, gnss_q=gnss_q, reward=reward)


def gen_obstacles(n_static=40, n_dynamic=4):
    rng = np.random.default_rng(7)
    static  = [{"position": rng.uniform([5,5,0], [190,190,0]) + [0,0,rng.uniform(5,25)],
                "radius": rng.uniform(3, 15), "type": "building"} for _ in range(n_static)]
    dynamic = [{"position": rng.uniform([20,20,8], [160,160,30]),
                "radius": rng.uniform(1, 3), "type": "dynamic"} for _ in range(n_dynamic)]
    return static, dynamic


def gen_sensor_data(n=1500):
    rng = np.random.default_rng(12)
    t = np.linspace(0, 60, n)

    true_pos = np.column_stack([
        50 + 30*np.sin(t*0.1),
        50 + 30*np.cos(t*0.08),
        20 + 5*np.sin(t*0.05),
    ])
    ekf_noise  = rng.normal(0, 0.2, (n, 3))
    ukf_noise  = rng.normal(0, 0.18, (n, 3))
    gnss_noise = rng.normal(0, 0.45, (n, 3))

    ekf_pos    = true_pos + ekf_noise
    ukf_pos    = true_pos + ukf_noise
    gnss_pos   = true_pos + gnss_noise

    ekf_err  = np.linalg.norm(ekf_pos  - true_pos, axis=1)
    ukf_err  = np.linalg.norm(ukf_pos  - true_pos, axis=1)
    gnss_err = np.linalg.norm(gnss_pos - true_pos, axis=1)

    return dict(t=t, true_pos=true_pos, ekf_pos=ekf_pos, ukf_pos=ukf_pos,
                gnss_pos=gnss_pos, ekf_err=ekf_err, ukf_err=ukf_err, gnss_err=gnss_err)


def gen_training_curve(n_eps=800):
    rng   = np.random.default_rng(99)
    eps   = np.arange(n_eps)
    trend = -150 + 280 * (1 - np.exp(-eps / 200))
    noise = rng.normal(0, 25, n_eps) * np.exp(-eps / 400) + rng.normal(0, 10, n_eps)
    return eps, trend + noise


def gen_lidar_scan():
    rng    = np.random.default_rng(5)
    angles = np.linspace(0, 2*math.pi, 360)
    ranges = 60 + 30 * np.sin(angles * 3 + 0.5) + rng.normal(0, 3, 360)
    # Insert close obstacles
    ranges[45:65]   = rng.uniform(5, 15, 20)
    ranges[200:220] = rng.uniform(8, 20, 20)
    ranges[310:330] = rng.uniform(12, 25, 20)
    return angles, np.clip(ranges, 2, 100)


def gen_swarm_positions(n_agents=5, n_steps=200):
    rng = np.random.default_rng(3)
    # V-formation moving forward
    formation_offset = np.array([
        [0, 0, 0], [-8, 8, -1], [8, 8, -1], [-16, 16, -2], [16, 16, -2]
    ])[:n_agents]
    leader_path = np.column_stack([
        np.linspace(20, 150, n_steps),
        np.linspace(20, 120, n_steps),
        np.full(n_steps, 15.0),
    ])
    positions = []
    for step in range(n_steps):
        step_pos = []
        for i in range(n_agents):
            pos = leader_path[step] + formation_offset[i] + rng.normal(0, 0.5, 3)
            step_pos.append(pos)
        positions.append(step_pos)
    return leader_path, positions


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: 3D Flight Trajectory
# ──────────────────────────────────────────────────────────────────────────────

def plot_trajectory_3d(traj, static_obs, dynamic_obs):
    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(16, 11))
        ax  = fig.add_subplot(111, projection="3d")
        xs, ys, zs = traj["xs"], traj["ys"], traj["zs"]
        speed      = traj["speed"]

        pts  = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        norm = Normalize(vmin=0, vmax=speed.max())
        lc   = Line3DCollection(segs, cmap="plasma", norm=norm, linewidth=2.5, alpha=0.95)
        lc.set_array(speed[:-1])
        ax.add_collection3d(lc)

        sm = ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.06, shrink=0.45, aspect=15)
        cb.set_label("Speed (m/s)", color="#a0c0e0", fontsize=10)
        cb.ax.yaxis.set_tick_params(color="#607090")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#607090")

        ax.scatter(xs[0], ys[0], zs[0], c=GREEN, s=200, zorder=5, marker="o",
                   label="Start", depthshade=False, edgecolors="white", linewidths=1.5)
        ax.scatter(xs[-1], ys[-1], zs[-1], c=YELLOW, s=300, zorder=5, marker="*",
                   label="End / Goal", depthshade=False)

        ax.plot(xs, ys, np.zeros_like(zs), "--", color="#ffffff15", linewidth=1.0, zorder=1)

        for obs in static_obs[:30]:
            p = obs["position"]
            r = obs["radius"]
            ax.scatter(p[0], p[1], p[2], c="#44444488", s=max(15, r*20),
                       marker="s", depthshade=True, alpha=0.6)

        for obs in dynamic_obs:
            p = obs["position"]
            ax.scatter(p[0], p[1], p[2], c="#ff335566", s=120, marker="^",
                       depthshade=True, alpha=0.8, label="Dynamic obstacle")

        ax.set_xlabel("East (m)", labelpad=12, fontsize=10)
        ax.set_ylabel("North (m)", labelpad=12, fontsize=10)
        ax.set_zlabel("Altitude (m)", labelpad=12, fontsize=10)
        ax.set_title("Autonomous UAV — 3D Flight Trajectory",
                     fontsize=15, color=ACCENT, pad=18, fontweight="bold")
        ax.legend(facecolor="#0d1b2e", edgecolor="#1e3050", labelcolor="#a0c0e0",
                  fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#1e3050")
        ax.yaxis.pane.set_edgecolor("#1e3050")
        ax.zaxis.pane.set_edgecolor("#1e3050")

        path = OUT_DIR / "01_trajectory_3d.png"
        plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ {path.name}")
        return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: Sensor Fusion Analysis
# ──────────────────────────────────────────────────────────────────────────────

def plot_sensor_fusion(sensor):
    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(18, 10))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
        fig.suptitle("EKF / UKF Sensor Fusion Analysis", fontsize=15,
                     color=ACCENT, fontweight="bold", y=0.97)

        t = sensor["t"]

        axes_data = [
            (gs[0,0], sensor["ekf_pos"][:,0], sensor["true_pos"][:,0], "Position X (m)"),
            (gs[0,1], sensor["ekf_pos"][:,1], sensor["true_pos"][:,1], "Position Y (m)"),
            (gs[0,2], sensor["ekf_pos"][:,2], sensor["true_pos"][:,2], "Altitude (m)"),
        ]
        for spec, fused, true, label in axes_data:
            ax = fig.add_subplot(spec)
            ax.plot(t, true,  color=GREEN,  linewidth=2.2, alpha=0.9, label="True")
            ax.plot(t, fused, color=ACCENT, linewidth=1.5, alpha=0.8, linestyle="--", label="EKF")
            ax.fill_between(t, true, fused, alpha=0.10, color=ACCENT)
            ax.set_ylabel(label, fontsize=9); ax.set_xlabel("Time (s)", fontsize=9)
            ax.legend(fontsize=8, facecolor="#0d1b2e", edgecolor="#1e3050", labelcolor="#a0c0e0")
            ax.grid(True, alpha=0.25)

        # Error comparison
        ax4 = fig.add_subplot(gs[1, 0])
        window = np.ones(30)/30
        ax4.plot(t, np.convolve(sensor["gnss_err"], window, "same"), color=ORANGE,
                 linewidth=2, label="GNSS raw")
        ax4.plot(t, np.convolve(sensor["ekf_err"],  window, "same"), color=ACCENT,
                 linewidth=2, label="EKF fused")
        ax4.plot(t, np.convolve(sensor["ukf_err"],  window, "same"), color=PURPLE,
                 linewidth=2, linestyle="--", label="UKF fused")
        ax4.fill_between(t, np.convolve(sensor["ekf_err"], window, "same"), alpha=0.15, color=ACCENT)
        ax4.set_title("Position Error (m)", fontsize=10, color=ACCENT)
        ax4.set_xlabel("Time (s)", fontsize=9)
        ax4.legend(fontsize=8, facecolor="#0d1b2e", edgecolor="#1e3050", labelcolor="#a0c0e0")
        ax4.grid(True, alpha=0.25)

        # RMSE bar
        ax5 = fig.add_subplot(gs[1, 1])
        methods = ["GNSS\nRaw", "EKF", "UKF", "Hybrid"]
        rmses   = [0.44, 0.23, 0.19, 0.21]
        colors  = [ORANGE, ACCENT, PURPLE, GREEN]
        bars = ax5.bar(methods, rmses, color=colors, alpha=0.85, width=0.5,
                        edgecolor=colors, linewidth=1.5)
        for bar, v in zip(bars, rmses):
            ax5.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.2f}m",
                     ha="center", va="bottom", fontsize=9, color=bar.get_facecolor())
        ax5.set_title("Position RMSE (m)", fontsize=10, color=ACCENT)
        ax5.set_ylabel("RMSE (m)", fontsize=9)
        ax5.grid(True, alpha=0.25, axis="y")

        # Covariance trace
        ax6 = fig.add_subplot(gs[1, 2])
        cov_trace = 0.5 * np.exp(-t / 15) + 0.05 + np.random.default_rng(11).normal(0, 0.005, len(t))
        ax6.plot(t, cov_trace, color=YELLOW, linewidth=2)
        ax6.fill_between(t, cov_trace, alpha=0.15, color=YELLOW)
        ax6.axhline(0.1, color=RED, linewidth=1.5, linestyle="--", label="Threshold")
        ax6.set_title("EKF Covariance Trace", fontsize=10, color=ACCENT)
        ax6.set_xlabel("Time (s)", fontsize=9)
        ax6.legend(fontsize=8, facecolor="#0d1b2e", edgecolor="#1e3050", labelcolor="#a0c0e0")
        ax6.grid(True, alpha=0.25)

        path = OUT_DIR / "02_sensor_fusion.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ {path.name}")
        return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: LiDAR & Obstacle Map
# ──────────────────────────────────────────────────────────────────────────────

def plot_lidar_obstacle_map(angles, ranges, traj):
    with plt.rc_context(DARK):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("LiDAR Obstacle Mapping", fontsize=14, color=ACCENT,
                     fontweight="bold")

        # Polar LiDAR
        ax1 = plt.subplot(121, projection="polar")
        ax1.set_facecolor("#0d1b2e")
        norm_r = ranges / ranges.max()
        cmap   = plt.cm.cool
        for i in range(len(angles)-1):
            c = cmap(norm_r[i])
            ax1.fill_between([angles[i], angles[i+1]], [0, 0],
                              [ranges[i], ranges[i+1]], color=c, alpha=0.7)
        ax1.plot(angles, ranges, color=ACCENT, linewidth=1.5, alpha=0.9)
        # Safety ring
        safety = np.full(360, 8.0)
        ax1.plot(np.linspace(0, 2*math.pi, 360), safety,
                 color=RED, linewidth=2, linestyle="--", alpha=0.8, label="Safety (8m)")
        ax1.set_title("360° LiDAR Scan", color=ACCENT, pad=15, fontsize=12)
        ax1.tick_params(colors="#607090")
        ax1.grid(color="#162030", alpha=0.8)
        ax1.set_facecolor("#0d1b2e")
        ax1.legend(loc="lower right", facecolor="#0d1b2e",
                   edgecolor="#1e3050", labelcolor="#a0c0e0", fontsize=9)

        # 2D obstacle + path map
        ax2 = axes[1]
        ax2.set_facecolor("#0d1b2e")
        xs, ys = traj["xs"], traj["ys"]

        # Heatmap
        h, xe, ye = np.histogram2d(xs, ys, bins=40, range=[[0,200],[0,200]])
        ax2.imshow(h.T, origin="lower", extent=[0,200,0,200],
                   cmap="hot", alpha=0.5, aspect="equal", interpolation="gaussian")

        # Trajectory
        speed  = traj["speed"]
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        norm2  = Normalize(vmin=0, vmax=speed.max())
        lc2    = LineCollection(segs, cmap="plasma", norm=norm2, linewidth=2, alpha=0.9)
        lc2.set_array(speed[:-1])
        ax2.add_collection(lc2)

        ax2.scatter(xs[0], ys[0], c=GREEN, s=200, zorder=5, marker="o", label="Start")
        ax2.scatter(xs[-1], ys[-1], c=YELLOW, s=250, zorder=5, marker="*", label="Goal")
        ax2.set_xlim(0, 200); ax2.set_ylim(0, 200)
        ax2.set_xlabel("East (m)", fontsize=10)
        ax2.set_ylabel("North (m)", fontsize=10)
        ax2.set_title("2D Path Heatmap + Trajectory", color=ACCENT, fontsize=12)
        ax2.legend(facecolor="#0d1b2e", edgecolor="#1e3050",
                   labelcolor="#a0c0e0", fontsize=9)
        ax2.grid(True, alpha=0.2)

        plt.tight_layout()
        path = OUT_DIR / "03_lidar_obstacle_map.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ {path.name}")
        return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4: RL Training Dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_dashboard(traj):
    eps_sac, rew_sac = gen_training_curve(800)
    eps_ppo, rew_ppo = gen_training_curve(800)
    rew_ppo -= 30; rew_ppo += np.random.default_rng(7).normal(0, 15, 800)
    eps_dqn, rew_dqn = gen_training_curve(800)
    rew_dqn -= 60; rew_dqn += np.random.default_rng(8).normal(0, 20, 800)

    window = np.ones(50) / 50
    ma_sac = np.convolve(rew_sac, window, "same")
    ma_ppo = np.convolve(rew_ppo, window, "same")
    ma_dqn = np.convolve(rew_dqn, window, "same")

    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(18, 12))
        gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
        fig.suptitle("Reinforcement Learning Training Dashboard",
                     fontsize=15, color=ACCENT, fontweight="bold")

        # Reward curves
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.fill_between(eps_sac, rew_sac, alpha=0.08, color=ACCENT)
        ax1.fill_between(eps_ppo, rew_ppo, alpha=0.08, color=GREEN)
        ax1.fill_between(eps_dqn, rew_dqn, alpha=0.08, color=ORANGE)
        ax1.plot(eps_sac, rew_sac, color=ACCENT+"44", linewidth=0.6)
        ax1.plot(eps_ppo, rew_ppo, color=GREEN+"44",  linewidth=0.6)
        ax1.plot(eps_dqn, rew_dqn, color=ORANGE+"44", linewidth=0.6)
        ax1.plot(eps_sac, ma_sac, color=ACCENT,  linewidth=2.5, label="SAC")
        ax1.plot(eps_ppo, ma_ppo, color=GREEN,   linewidth=2.5, label="PPO")
        ax1.plot(eps_dqn, ma_dqn, color=ORANGE,  linewidth=2.5, label="DQN")
        ax1.axhline(0, color="#607090", linewidth=0.8, linestyle="--")
        ax1.set_title("Episode Reward — SAC vs PPO vs DQN", color=ACCENT, fontsize=11)
        ax1.set_xlabel("Episode", fontsize=9); ax1.set_ylabel("Reward", fontsize=9)
        ax1.legend(facecolor="#0d1b2e", edgecolor="#1e3050", labelcolor="#a0c0e0")
        ax1.grid(True, alpha=0.2)

        # Success rate over training
        ax2 = fig.add_subplot(gs[0, 2])
        eps_q = np.linspace(0, 800, 100)
        sr_sac = 0.87 * (1 - np.exp(-eps_q / 200)) + np.random.default_rng(1).normal(0, 0.02, 100)
        sr_ppo = 0.72 * (1 - np.exp(-eps_q / 250)) + np.random.default_rng(2).normal(0, 0.02, 100)
        sr_dqn = 0.58 * (1 - np.exp(-eps_q / 350)) + np.random.default_rng(3).normal(0, 0.02, 100)
        ax2.plot(eps_q, np.clip(sr_sac, 0, 1), color=ACCENT, linewidth=2.2, label="SAC")
        ax2.plot(eps_q, np.clip(sr_ppo, 0, 1), color=GREEN, linewidth=2.2, label="PPO")
        ax2.plot(eps_q, np.clip(sr_dqn, 0, 1), color=ORANGE, linewidth=2.2, label="DQN")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Success Rate", color=ACCENT, fontsize=11)
        ax2.set_xlabel("Episode"); ax2.set_ylabel("Rate")
        ax2.legend(facecolor="#0d1b2e", edgecolor="#1e3050", labelcolor="#a0c0e0", fontsize=8)
        ax2.grid(True, alpha=0.2)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        # Telemetry over flight time
        t = traj["t"]
        panels = [
            (gs[1,0], traj["zs"],      "Altitude (m)",     ACCENT),
            (gs[1,1], traj["speed"],   "Speed (m/s)",       ORANGE),
            (gs[1,2], traj["battery"], "Battery (%)",       YELLOW),
            (gs[2,0], traj["gnss_q"],  "GNSS Quality",      GREEN),
            (gs[2,1], np.gradient(traj["reward"]), "Reward Signal", PURPLE),
        ]
        for spec, vals, label, clr in panels:
            ax = fig.add_subplot(spec)
            ax.plot(t[:len(vals)], vals, color=clr, linewidth=1.8, alpha=0.9)
            ax.fill_between(t[:len(vals)], vals, alpha=0.12, color=clr)
            ax.set_title(label, fontsize=9, color=clr)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.grid(True, alpha=0.2)

        # Benchmark bar chart
        ax_bar = fig.add_subplot(gs[2, 2])
        scenarios = ["Nominal", "GNSS\nDrop", "High\nWind", "Dense\nObs"]
        sac_succ  = [0.873, 0.712, 0.685, 0.598]
        x         = np.arange(len(scenarios))
        bars = ax_bar.bar(x, sac_succ, width=0.5, color=ACCENT,
                           alpha=0.85, edgecolor=ACCENT, linewidth=1.5)
        for bar, v in zip(bars, sac_succ):
            ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.0%}",
                        ha="center", va="bottom", fontsize=8.5, color=ACCENT)
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(scenarios, fontsize=8)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_title("Robustness — SAC", color=ACCENT, fontsize=10)
        ax_bar.set_ylabel("Success Rate")
        ax_bar.grid(True, alpha=0.2, axis="y")
        ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

        path = OUT_DIR / "04_rl_training_dashboard.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ {path.name}")
        return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5: Swarm Intelligence
# ──────────────────────────────────────────────────────────────────────────────

def plot_swarm(leader_path, swarm_positions):
    rng = np.random.default_rng(42)
    n_agents = len(swarm_positions[0])
    agent_colors = [ACCENT, GREEN, YELLOW, ORANGE, PURPLE]

    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(16, 8))
        gs  = GridSpec(1, 2, figure=fig, wspace=0.30)
        fig.suptitle("Swarm Intelligence — V-Formation Flight",
                     fontsize=14, color=ACCENT, fontweight="bold")

        # 2D Formation plot
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor("#0d1b2e")
        n_steps = len(swarm_positions)
        tail = 30

        for i in range(n_agents):
            traj_x = [swarm_positions[s][i][0] for s in range(n_steps)]
            traj_y = [swarm_positions[s][i][1] for s in range(n_steps)]
            ax1.plot(traj_x, traj_y, color=agent_colors[i], linewidth=1.8,
                     alpha=0.6, linestyle="--")
            ax1.scatter(traj_x[-1], traj_y[-1], c=agent_colors[i], s=150, zorder=5,
                        edgecolors="white", linewidths=1.2,
                        label=f"UAV {i+1}" + (" (Leader)" if i==0 else ""))
            # Current tail
            ax1.plot(traj_x[-tail:], traj_y[-tail:], color=agent_colors[i],
                     linewidth=2.5, alpha=0.9)

        # Communication links
        last_pos = [swarm_positions[-1][i] for i in range(n_agents)]
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                dist = np.linalg.norm(np.array(last_pos[i]) - np.array(last_pos[j]))
                if dist < 40:
                    ax1.plot([last_pos[i][0], last_pos[j][0]],
                              [last_pos[i][1], last_pos[j][1]],
                              color="#ffffff22", linewidth=1.0, linestyle=":")

        ax1.set_xlim(0, 180); ax1.set_ylim(0, 180)
        ax1.set_xlabel("East (m)", fontsize=10); ax1.set_ylabel("North (m)", fontsize=10)
        ax1.set_title("2D Swarm Trajectories", color=ACCENT, fontsize=11)
        ax1.legend(facecolor="#0d1b2e", edgecolor="#1e3050",
                   labelcolor="#a0c0e0", fontsize=8, loc="upper left")
        ax1.grid(True, alpha=0.2)

        # 3D swarm
        ax3 = fig.add_subplot(gs[1], projection="3d")
        ax3.set_facecolor("#0d1b2e")
        for i in range(n_agents):
            traj_x = [swarm_positions[s][i][0] for s in range(n_steps)]
            traj_y = [swarm_positions[s][i][1] for s in range(n_steps)]
            traj_z = [swarm_positions[s][i][2] + i*1.5 for s in range(n_steps)]
            ax3.plot(traj_x, traj_y, traj_z, color=agent_colors[i],
                     linewidth=1.8, alpha=0.7)
            ax3.scatter(traj_x[-1], traj_y[-1], traj_z[-1],
                        c=agent_colors[i], s=120, zorder=5, depthshade=False)

        ax3.set_title("3D Formation Flight", color=ACCENT, fontsize=11)
        ax3.set_xlabel("E (m)", labelpad=8); ax3.set_ylabel("N (m)", labelpad=8)
        ax3.set_zlabel("Alt (m)", labelpad=8)
        ax3.xaxis.pane.fill = False; ax3.yaxis.pane.fill = False; ax3.zaxis.pane.fill = False
        ax3.xaxis.pane.set_edgecolor("#1e3050")
        ax3.yaxis.pane.set_edgecolor("#1e3050")
        ax3.zaxis.pane.set_edgecolor("#1e3050")
        ax3.grid(True, alpha=0.2)

        path = OUT_DIR / "05_swarm_intelligence.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ {path.name}")
        return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  AUTONOMOUS UAV NAVIGATION SYSTEM")
    print("  Demo Visualization Generator")
    print("═"*60)
    print(f"\nOutput directory: {OUT_DIR.resolve()}\n")

    print("Generating synthetic data...")
    traj      = gen_trajectory(n=1500, env="urban")
    static, dynamic = gen_obstacles()
    sensor    = gen_sensor_data(n=1500)
    angles, ranges = gen_lidar_scan()
    leader_path, swarm_pos = gen_swarm_positions(n_agents=5, n_steps=200)

    print("\nRendering plots:")
    paths = []
    paths.append(plot_trajectory_3d(traj, static, dynamic))
    paths.append(plot_sensor_fusion(sensor))
    paths.append(plot_lidar_obstacle_map(angles, ranges, traj))
    paths.append(plot_training_dashboard(traj))
    paths.append(plot_swarm(leader_path, swarm_pos))

    print("\n" + "─"*60)
    print(f"  ✅  {len(paths)} plots saved to: {OUT_DIR.resolve()}")
    print("─"*60 + "\n")


if __name__ == "__main__":
    main()
