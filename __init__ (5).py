"""
=============================================================================
scripts/evaluate.py  –  RL Agent Evaluation & Benchmarking
=============================================================================
Evaluates trained agents across multiple environments and metrics:
  - Success rate (goal reached)
  - Collision rate
  - Path efficiency (ratio vs optimal)
  - Energy consumption
  - GNSS degradation robustness
  - Dynamic obstacle avoidance

Usage:
    python scripts/evaluate.py --model models/final_model --episodes 100
=============================================================================
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from envs.uav_env import UAVNavigationEnv

log = get_logger("EVAL")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    model,
    env: UAVNavigationEnv,
    n_episodes: int = 50,
    config: dict = None,
    render: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation suite and return metrics dict.
    """
    metrics = {
        "success_rate":       0.0,
        "collision_rate":     0.0,
        "timeout_rate":       0.0,
        "mean_reward":        0.0,
        "mean_ep_length":     0.0,
        "mean_battery_used":  0.0,
        "mean_path_length_m": 0.0,
        "mean_dist_to_goal":  0.0,
        "n_episodes":         n_episodes,
    }

    episode_rewards  = []
    episode_lengths  = []
    successes        = []
    collisions       = []
    battery_used     = []
    path_lengths     = []
    final_dists      = []

    log.info(f"Evaluating over {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        ep_len    = 0
        done      = False
        prev_pos  = env._dynamics.pos.copy()
        path_len  = 0.0
        start_bat = env._dynamics.battery_pct

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # Random baseline
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            ep_len    += 1

            cur_pos = env._dynamics.pos.copy()
            path_len += float(np.linalg.norm(cur_pos - prev_pos))
            prev_pos  = cur_pos

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        successes.append(1 if info.get("distance_to_goal", 999) < 3.0 else 0)
        collisions.append(1 if info.get("collision", False) else 0)
        battery_used.append(start_bat - env._dynamics.battery_pct)
        path_lengths.append(path_len)
        final_dists.append(info.get("distance_to_goal", 0.0))

        if (ep + 1) % 10 == 0:
            log.info(f"  Episode {ep+1}/{n_episodes} | "
                     f"success={np.mean(successes):.2%} | "
                     f"mean_reward={np.mean(episode_rewards):.1f}")

    metrics["success_rate"]       = float(np.mean(successes))
    metrics["collision_rate"]     = float(np.mean(collisions))
    metrics["timeout_rate"]       = float(1.0 - np.mean(successes) - np.mean(collisions))
    metrics["mean_reward"]        = float(np.mean(episode_rewards))
    metrics["std_reward"]         = float(np.std(episode_rewards))
    metrics["mean_ep_length"]     = float(np.mean(episode_lengths))
    metrics["mean_battery_used"]  = float(np.mean(battery_used))
    metrics["mean_path_length_m"] = float(np.mean(path_lengths))
    metrics["mean_dist_to_goal"]  = float(np.mean(final_dists))

    _print_report(metrics)
    return metrics


def _print_report(m: dict) -> None:
    sep = "═" * 52
    log.info(f"\n{sep}")
    log.info("  EVALUATION REPORT")
    log.info(sep)
    log.info(f"  Episodes           : {m['n_episodes']}")
    log.info(f"  Success Rate       : {m['success_rate']:.1%}")
    log.info(f"  Collision Rate     : {m['collision_rate']:.1%}")
    log.info(f"  Timeout Rate       : {m['timeout_rate']:.1%}")
    log.info(f"  Mean Reward        : {m['mean_reward']:.2f} ± {m.get('std_reward',0):.2f}")
    log.info(f"  Mean Ep Length     : {m['mean_ep_length']:.0f} steps")
    log.info(f"  Mean Path Length   : {m['mean_path_length_m']:.1f} m")
    log.info(f"  Mean Battery Used  : {m['mean_battery_used']:.1f}%")
    log.info(f"  Mean Dist to Goal  : {m['mean_dist_to_goal']:.1f} m")
    log.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Robustness Benchmarks
# ──────────────────────────────────────────────────────────────────────────────

def robustness_benchmark(model, config: dict, n_episodes: int = 20) -> Dict[str, Any]:
    """
    Test agent robustness under degraded conditions:
      - GNSS dropout
      - High wind
      - Dense obstacles
      - Low battery start
    """
    results = {}
    scenarios = [
        ("nominal",        {}),
        ("gnss_dropout",   {"sensors.gnss.dropout_probability": 0.05}),
        ("high_wind",      {"environment.wind.base_speed_ms": 8.0,
                            "environment.wind.turbulence_intensity": 0.8}),
        ("dense_obs",      {"environment.obstacles.static_count": 100}),
    ]

    for name, overrides in scenarios:
        log.info(f"\nBenchmark: {name}")
        cfg = ConfigLoader()
        for k, v in overrides.items():
            cfg.patch(k, v)

        env = UAVNavigationEnv(cfg.raw)
        m   = run_evaluation(model, env, n_episodes=n_episodes, config=cfg.raw)
        results[name] = {
            "success_rate":   m["success_rate"],
            "collision_rate": m["collision_rate"],
            "mean_reward":    m["mean_reward"],
        }
        env.close()

    log.info("\n── Robustness Summary ────────────────────────────")
    for name, r in results.items():
        log.info(f"  {name:<20s}: success={r['success_rate']:.1%}  "
                 f"collision={r['collision_rate']:.1%}  "
                 f"reward={r['mean_reward']:.1f}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UAV Agent Evaluation")
    p.add_argument("--model",      default=None,          help="Path to saved model")
    p.add_argument("--algo",       default="SAC",         choices=["SAC", "PPO", "DQN"])
    p.add_argument("--episodes",   default=50,            type=int)
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--env-type",   default="urban")
    p.add_argument("--robustness", action="store_true",   help="Run robustness benchmarks")
    p.add_argument("--output",     default="evaluation/results.json")
    return p.parse_args()


def main():
    args = parse_args()

    cfg_loader = ConfigLoader(args.config)
    cfg_loader.patch("environment.type", args.env_type)
    config = cfg_loader.raw

    env   = UAVNavigationEnv(config, record=False)
    model = None

    if args.model:
        from agents.rl_agents import RLTrainingManager
        trainer = RLTrainingManager(config)
        model   = trainer.load(args.model, env)

    metrics = run_evaluation(model, env, n_episodes=args.episodes, config=config)

    if args.robustness:
        rob = robustness_benchmark(model, config, n_episodes=20)
        metrics["robustness"] = rob

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Results saved → {out_path}")

    env.close()


if __name__ == "__main__":
    main()
