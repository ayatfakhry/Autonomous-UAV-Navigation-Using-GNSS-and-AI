"""
=============================================================================
scripts/train.py  –  RL Training Pipeline
=============================================================================
Train SAC / PPO / DQN agents on the UAV navigation environment.

Usage:
    python scripts/train.py --algo SAC --timesteps 2000000
    python scripts/train.py --algo PPO --timesteps 1000000 --eval
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from envs.uav_env import UAVNavigationEnv
from agents.rl_agents import RLTrainingManager

log = get_logger("TRAINING")


def parse_args():
    p = argparse.ArgumentParser(description="UAV RL Training")
    p.add_argument("--algo",       default="SAC",          choices=["SAC","PPO","DQN"])
    p.add_argument("--timesteps",  default=2_000_000,      type=int)
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--env-type",   default="urban",
                   choices=["urban","forest","mountain","open_field"])
    p.add_argument("--seed",       default=42,             type=int)
    p.add_argument("--eval",       action="store_true",    help="Run evaluation after training")
    p.add_argument("--resume",     default=None,           help="Path to resume from checkpoint")
    p.add_argument("--n-envs",     default=1,              type=int, help="Parallel envs (SB3)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg_loader = ConfigLoader(args.config)
    cfg_loader.patch("rl.algorithm", args.algo)
    cfg_loader.patch("rl.training.total_timesteps", args.timesteps)
    cfg_loader.patch("environment.type", args.env_type)
    config = cfg_loader.raw

    log.info(f"Training config: algo={args.algo}, timesteps={args.timesteps:,}, env={args.env_type}")

    # Create environment
    env = UAVNavigationEnv(config, record=True)

    # Build & train agent
    trainer = RLTrainingManager(config)
    model   = trainer.build(env)

    if args.resume:
        log.info(f"Resuming from: {args.resume}")
        trainer.load(args.resume, env)

    try:
        trainer.train()
    except KeyboardInterrupt:
        log.warning("Training interrupted — saving checkpoint")
        trainer.save("models/interrupted_checkpoint")

    if args.eval:
        log.info("Running evaluation...")
        from scripts.evaluate import run_evaluation
        run_evaluation(model, env, n_episodes=20, config=config)


if __name__ == "__main__":
    main()
